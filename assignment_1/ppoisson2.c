/*
 * SEQ_Poisson.c
 * 2D Poison equation solver
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define DEBUG 1

#define max(a, b) ((a) > (b) ? a : b)

enum
{
  X_DIR,
  Y_DIR
};

/* global variables */
int gridsize[2];
double precision_goal = 0.0001; /* precision_goal of solution */
int max_iter = 5000;            /* maximum number of iterations alowed */
MPI_Datatype border_type[2];    /* Datatypes for vertical and horizontal exchange */
int *gridsizes;
int grid_length = 1;
int grid_size_idx;
// char fn_template[] = "%s/procg=%ix%i__gs=%ix%i_wl=%3.2f_wh=%3.2f_nomega=%i_swpl=%i_swph=%i_eloop=%i_%s.dat";

/* process specific variables */
int proc_rank;                                    /* process rank and number of ranks */
int proc_coord[2];                                /* coordinates of current process in processgrid */
int proc_top, proc_right, proc_bottom, proc_left; /* ranks of neigboring procs        */
int offset[2];                                    /* offset of subgrid handled by current process */

int P;              /* total number of processes */
int P_grid[2];      /* process grid dimensions        */
MPI_Comm grid_comm; /* grid COMMUNICATOR        */
MPI_Status status;

/* benchmark related variables */
clock_t ticks;    /* number of systemticks */
double *cpu_util; /* CPU utilization */
int timer_on = 0; /* is timer running? */
double wtime;
double *wtimes;
int count;
int *iters;
int current_iter;
double wtime_sum;

/* local grid related variables */
double **phi; /* grid */
int **source; /* TRUE if subgrid element is a source */
int dim[2];   /* grid dimensions */

/* toggles */
int benchmark_flag = 0;
int error_flag = 0;
int track_errors = 0;
int write_output_flag = 0;
int efficient_loop_flag = 1;
int latency_flag = 0;

/* relaxation paramater */
double omega;
double *omegas;
// omegas = malloc(sizeof(double));
// omegas[0] = 1.95;
int omega_length = 1;

/* error array*/
double *errors;

/* sweep array */
int sweep;
int *sweeps;
// sweeps = malloc(sizeof(int));
// sweeps[0] = 1;
int **iters_sweep_vs_omega;
double **times_sweep_vs_omega;
int sweep_length = 1;

/* latency analysis */
double latency;
double *latencies;
double byte;
double *bytes;
int latency_length;

/*time v iterations*/
double* time_by_iteration;
int timeviter_flag;
int time_by_iteration_size = 0;


/* function declarations */
void Setup_Grid();
void Setup_Proc_Grid(int argc, char **argv);
void Get_CLIs(int argc, char **argv);
void Setup_MPI_Datatypes();
void Exchange_Borders();
double Do_Step(int parity);
void Solve();
void Write_Grid();
void Benchmark();
void Error_Analysis();
void Sweep_Analysis();
void Latency_Analysis();
void timeVIteration();
void Clean_Up_Problemdata();
void Clean_Up_Metadata();
void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();
void generate_fn(char *fn, char *folder, char *type);

void start_timer()
{
  if (!timer_on)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    ticks = clock();
    wtime = MPI_Wtime();
    timer_on = 1;
  }
}

void resume_timer()
{
  if (!timer_on)
  {
    ticks = clock() - ticks;
    wtime = MPI_Wtime() - wtime;
    timer_on = 1;
  }
}

void stop_timer()
{
  if (timer_on)
  {
    ticks = clock() - ticks;
    wtime = MPI_Wtime() - wtime;
    timer_on = 0;
  }
}

void print_timer()
{
  if (timer_on)
  {
    stop_timer();
    printf("(%i) Elapsed Wtime %14.6f s (%5.1f%% CPU)\n",
           proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
    resume_timer();
  }
  else
  {
    printf("(%i) Elapsed Wtime %14.6f s (%5.1f%% CPU)\n",
           proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
  }
}

void generate_fn(char *fn, char *folder, char *type)
{
  char fn_template[] = "%s/procg=%ix%i__gs=%ix%i_wl=%3.2f_wh=%3.2f_nomega=%i_swpl=%i_swph=%i_eloop=%i_%s.dat";
  sprintf(fn, fn_template, folder, P_grid[X_DIR], P_grid[Y_DIR], gridsize[X_DIR],
          gridsize[Y_DIR], omegas[0], omegas[omega_length - 1], omega_length,
          sweeps[0], sweeps[sweep_length - 1], efficient_loop_flag, type);
}

void Debug(char *mesg, int terminate)
{
  if (DEBUG || terminate)
    printf("%s\n", mesg);
  if (terminate)
    exit(1);
}

void Setup_Grid()
{
  int x, y, s;
  double source_x, source_y, source_val;
  int upper_offset[2];
  FILE *f;

  // Debug("Setup_Subgrid", 0);

  if (proc_rank == 0)
  {
    f = fopen("input.dat", "r");
    if (f == NULL)
      Debug("Error opening input.dat", 1);
    fscanf(f, "nx: %i\n", &gridsize[X_DIR]);
    fscanf(f, "ny: %i\n", &gridsize[Y_DIR]);
    fscanf(f, "precision goal: %lf\n", &precision_goal);
    fscanf(f, "max iterations: %i\n", &max_iter);
  }

  gridsize[X_DIR] = gridsizes[grid_size_idx];
  gridsize[Y_DIR] = gridsizes[grid_size_idx];

  // broadcast gridsize, precision_goal and max_iter to all processes
  MPI_Barrier(grid_comm);
  MPI_Bcast(&gridsize, 2, MPI_INT, 0, grid_comm);
  MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, grid_comm);
  MPI_Bcast(&max_iter, 1, MPI_INT, 0, grid_comm);
  MPI_Barrier(grid_comm);

  /* Calculate top  left  corner  coordinates  of  local  grid  */
  offset[X_DIR] = gridsize[X_DIR] * proc_coord[X_DIR] / P_grid[X_DIR];
  offset[Y_DIR] = gridsize[Y_DIR] * proc_coord[Y_DIR] / P_grid[Y_DIR];
  upper_offset[X_DIR] = gridsize[X_DIR] * (proc_coord[X_DIR] + 1) / P_grid[X_DIR];
  upper_offset[Y_DIR] = gridsize[Y_DIR] * (proc_coord[Y_DIR] + 1) / P_grid[Y_DIR];

  /* Calculate dimensions of  local  grid  */
  dim[Y_DIR] = upper_offset[Y_DIR] - offset[Y_DIR];
  dim[X_DIR] = upper_offset[X_DIR] - offset[X_DIR];

  /* Add space for rows/columns of neighboring grid */
  dim[Y_DIR] += 2;
  dim[X_DIR] += 2;

  /* allocate memory */
  if ((phi = malloc(dim[X_DIR] * sizeof(*phi))) == NULL)
    Debug("Setup_Subgrid : malloc(phi) failed", 1);
  if ((source = malloc(dim[X_DIR] * sizeof(*source))) == NULL)
    Debug("Setup_Subgrid : malloc(source) failed", 1);
  if ((phi[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**phi))) == NULL)
    Debug("Setup_Subgrid : malloc(*phi) failed", 1);
  if ((source[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**source))) == NULL)
    Debug("Setup_Subgrid : malloc(*source) failed", 1);
  for (x = 1; x < dim[X_DIR]; x++)
  {
    phi[x] = phi[0] + x * dim[Y_DIR];
    source[x] = source[0] + x * dim[Y_DIR];
  }

  /* set all values to '0' */
  for (x = 0; x < dim[X_DIR]; x++)
    for (y = 0; y < dim[Y_DIR]; y++)
    {
      phi[x][y] = 0.0;
      source[x][y] = 0;
    }

  /* put sources in field */
  MPI_Barrier(grid_comm);
  do
  {
    if (proc_rank == 0)
    {
      s = fscanf(f, "source: %lf %lf %lf\n", &source_x, &source_y, &source_val);
    }
    MPI_Bcast(&s, 1, MPI_INT, 0, grid_comm);
    if (s == 3)
    {
      MPI_Bcast(&source_x, 1, MPI_DOUBLE, 0, grid_comm);
      MPI_Bcast(&source_y, 1, MPI_DOUBLE, 0, grid_comm);
      MPI_Bcast(&source_val, 1, MPI_DOUBLE, 0, grid_comm);
      x = source_x * gridsize[X_DIR];
      y = source_y * gridsize[Y_DIR];
      x += 1;
      y += 1;
      x = x - offset[X_DIR];
      y = y - offset[Y_DIR];
      if (x > 0 && x < dim[X_DIR] - 1 && y > 0 && y < dim[Y_DIR] - 1)
      { /* indices in domain of this process */
        phi[x][y] = source_val;
        source[x][y] = 1;
      }
    }
  } while (s == 3);
  MPI_Barrier(grid_comm);

  if (proc_rank == 0)
  {
    fclose(f);
  }
}

void Setup_Proc_Grid(int argc, char **argv)
{
  int wrap_around[2];
  int reorder;
  // Debug("My_MPI_Init", 0);

  /* Retrieve the number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &P); /* find out how many processes there are        */

  /* Calculate the number of processes per column and per row for the grid */
  if (argc > 2)
  {
    P_grid[X_DIR] = atoi(argv[1]);
    P_grid[Y_DIR] = atoi(argv[2]);
    if (P_grid[X_DIR] * P_grid[Y_DIR] != P)
      Debug("ERROR Proces grid dimensions do not match with P ", 1);
  }
  else
  {
    Debug("ERROR Wrong parameter input", 1);
  }

  /* Create process topology (2D grid) */
  wrap_around[X_DIR] = 0;
  wrap_around[Y_DIR] = 0; /*  do  not  connect  first  and last process        */

  reorder = 1; /*  reorder process ranks        */

  /* Creates a new communicator grid_comm  */
  MPI_Cart_create(MPI_COMM_WORLD, 2, P_grid, wrap_around, reorder, &grid_comm);

  /* Retrieve new rank and cartesian coordinates of this process */
  MPI_Comm_rank(grid_comm, &proc_rank);                 /*  Rank  of  process  in  new  communicator        */
  MPI_Cart_coords(grid_comm, proc_rank, 2, proc_coord); /* Coordinates of process in new communicator */

  printf("(%i) (x,y)=(%i,%i)\n", proc_rank, proc_coord[X_DIR], proc_coord[Y_DIR]);

  /* calculate ranks of neighboring processes */
  MPI_Cart_shift(grid_comm, X_DIR, 1, &proc_left, &proc_right);

  /*  rank of processes proc_top and proc_bottom  */
  MPI_Cart_shift(grid_comm, Y_DIR, -1, &proc_bottom, &proc_top);

  if (DEBUG)
    printf("(%i) top %i,  right  %i,  bottom  %i,  left  %i\n", proc_rank, proc_top,
           proc_right, proc_bottom, proc_left);
}

void Get_CLIs(int argc, char **argv)
{
  int l, i;
  l = 0;
  double omega_start, omega_end, omega_step;
  int grid_start, grid_end, grid_step;
  int sweep_start, sweep_end, sweep_step;

  // default values (in case no CLI args are specified)
  gridsizes = malloc(sizeof(int));
  gridsizes[0] = 100;
  grid_length = 1;
  omegas = malloc(sizeof(double));
  omegas[0] = 1.95;
  omega_length = 1;
  sweeps = malloc(sizeof(int));
  sweeps[0] = 1;
  sweep_length = 1;

  if (argc > 3)
  {
    while (l < argc)
    {
      if (strcmp(argv[l], "-omega") == 0)
      {
        printf("(%i) Using omega value from command line\n", proc_rank);
        omegas = malloc(sizeof(double));
        omegas[0] = atof(argv[4]);
        omega_length = 1;
      }

      if (strcmp(argv[l], "-omegas") == 0)
      {
        printf("(%i) Using omega values from command line\n", proc_rank);
        omega_start = atof(argv[l + 1]);
        omega_end = atof(argv[l + 2]);
        omega_step = atof(argv[l + 3]);
        omega_length = (int)((omega_end - omega_start) / omega_step) + 1;
        if (omega_start < 0.0 || omega_start > 2.0 || omega_end < 0.0 || omega_end > 2.0 || omega_step < 0.0 || omega_step > 2.0)
          Debug("ERROR Omega values outside range [0,2]", 1);
        omegas = malloc(omega_length * sizeof(double));
        for (i = 0; i < omega_length; i++)
        {
          omegas[i] = omega_start + (double)i * omega_step;
        }
      }

      if (strcmp(argv[l], "-grid") == 0)
      {
        printf("(%i) Using grid size from command line\n", proc_rank);
        gridsizes = malloc(sizeof(int));
        gridsizes[0] = atoi(argv[l + 1]);
        grid_length = 1;
      }

      if (strcmp(argv[l], "-grids") == 0)
      {
        printf("(%i) Using grid sizes from command line\n", proc_rank);
        grid_start = atoi(argv[l + 1]);
        grid_end = atoi(argv[l + 2]);
        grid_step = atoi(argv[l + 3]);
        grid_length = ((grid_end - grid_start) / grid_step) + 1;
        if (grid_start < 0 || grid_end < 0 || grid_step < 0)
          Debug("ERROR Grid values outside range [0,inf]", 1);
        gridsizes = malloc(grid_length * sizeof(int));
        for (i = 0; i < grid_length; i++)
        {
          gridsizes[i] = grid_start + i * grid_step;
        }
      }

      if (strcmp(argv[l], "-output") == 0)
      {
        if (strcmp(argv[l + 1], "true") == 0)
        {
          printf("(%i) Writing output to file\n", proc_rank);
          write_output_flag = 1;
        }
        else if (strcmp(argv[l + 1], "false") == 0)
        {
          printf("(%i) Not writing output to file\n", proc_rank);
          write_output_flag = 0;
        }
        else
        {
          printf("(%i) Invalid output flag, no output will be generated\n", proc_rank);
          write_output_flag = 0;
        }
      }

      if (strcmp(argv[l], "-errors") == 0)
      {
        if (strcmp(argv[l + 1], "true") == 0)
        {
          printf("(%i) Tracking errors\n", proc_rank);
          track_errors = 1;
        }
        else if (strcmp(argv[l + 1], "false") == 0)
        {
          printf("(%i) Not tracking errors\n", proc_rank);
          track_errors = 0;
        }
        else
        {
          printf("(%i) Invalid errors flag, no errors will be tracked\n", proc_rank);
          track_errors = 0;
        }
      }

      if (strcmp(argv[l], "-benchmark") == 0)
      {
        if (strcmp(argv[l + 1], "true") == 0)
        {
          printf("(%i) Benchmarking\n", proc_rank);
          benchmark_flag = 1;
        }
        else if (strcmp(argv[l + 1], "false") == 0)
        {
          printf("(%i) Not benchmarking\n", proc_rank);
          benchmark_flag = 0;
        }
        else
        {
          printf("(%i) Invalid benchmarking flag, no benchmarking will be done\n", proc_rank);
          benchmark_flag = 0;
        }
      }

      if (strcmp(argv[l], "-sweeps") == 0)
      {
        printf("(%i) Using sweep values from command line\n", proc_rank);
        sweep_start = atoi(argv[l + 1]);
        sweep_end = atoi(argv[l + 2]);
        sweep_step = atoi(argv[l + 3]);
        sweep_length = (int)((sweep_end - sweep_start) / sweep_step) + 1;
        sweeps = malloc(sweep_length * sizeof(int));
        if (sweep_start < 0 || sweep_end < 0 || sweep_step < 0)
        {
          Debug("ERROR Sweep values outside range [0,inf]", 1);
        }
        for (i = 0; i < sweep_length; i++)
        {
          sweeps[i] = sweep_start + i * sweep_step;
        }
      }

      if (strcmp(argv[l], "-efficient-loop") == 0)
      {
        if (strcmp(argv[l + 1], "true") == 0)
        {
          printf("(%i) Using efficient loop\n", proc_rank);
          efficient_loop_flag = 1;
        }
        else if (strcmp(argv[l + 1], "false") == 0)
        {
          printf("(%i) Using inefficient loop\n", proc_rank);
          efficient_loop_flag = 0;
        }
        else
        {
          printf("(%i) Invalid efficient loop flag, using efficient loop\n", proc_rank);
          efficient_loop_flag = 1;
        }
      }

      if (strcmp(argv[l], "-latency") == 0)
      {
        if (strcmp(argv[l + 1], "true") == 0)
        {
          printf("(%i) Latency analysis\n", proc_rank);
          latency_flag = 1;
        }
        else if (strcmp(argv[l + 1], "false") == 0)
        {
          printf("(%i) Not doing latency analysis\n", proc_rank);
          latency_flag = 0;
        }
        else
        {
          printf("(%i) Invalid latency flag, no latency analysis will be done\n", proc_rank);
          latency_flag = 0;
        }
      }
      
      if (strcmp(argv[l], "-timeviter") == 0)
      {
        if (strcmp(argv[l + 1], "true") == 0)
        {
          printf("(%i) Time per iteration \n", proc_rank);
          timeviter_flag = 1;
        }
        else if (strcmp(argv[l + 1], "false") == 0)
        {
          printf("(%i) No time per iteration\n", proc_rank);
          timeviter_flag = 0;
        }
        else
        {
          printf("(%i) Invalid timeviter flag, using default\n", proc_rank);
          timeviter_flag = 0;
        }
      }

      l++;
    }
  }
  else
  {
    printf("(%i) No CLI args specified, using default values for grid size and omega\n", proc_rank);
  }
}

double Do_Step(int parity)
{
  int x, y;
  double old_phi;
  double max_err = 0.0;
  int x_parity;

  /* calculate interior of grid */
  if (efficient_loop_flag)
  {
    for (x = 1; x < dim[X_DIR] - 1; x++)
    {
      x_parity = (x + offset[X_DIR] + offset[Y_DIR] + parity) % 2;
      for (y = 1 + x_parity; y < dim[Y_DIR] - 1; y += 2)
      {
        // if ((offset[X_DIR] + x + offset[Y_DIR] + y) % 2 == parity && source[x][y] != 1)
        if (source[x][y] != 1)
        {
          old_phi = phi[x][y];
          phi[x][y] = (1 - omega) * phi[x][y] + omega * (phi[x + 1][y] + phi[x - 1][y] + phi[x][y + 1] + phi[x][y - 1]) * 0.25;
          if (max_err < fabs(old_phi - phi[x][y]))
            max_err = fabs(old_phi - phi[x][y]);
        }
      }
    }
  }
  else // use inefficient/naive loop over all grid points
  {
    for (x = 1; x < dim[X_DIR] - 1; x++)
    {
      for (y = 1; y < dim[Y_DIR] - 1; y++)
      {
        if ((offset[X_DIR] + x + offset[Y_DIR] + y) % 2 == parity && source[x][y] != 1)
        {
          old_phi = phi[x][y];
          phi[x][y] = (1 - omega) * phi[x][y] + omega * (phi[x + 1][y] + phi[x - 1][y] + phi[x][y + 1] + phi[x][y - 1]) * 0.25;
          if (max_err < fabs(old_phi - phi[x][y]))
            max_err = fabs(old_phi - phi[x][y]);
        }
      }
    }
  }

  return max_err;
}

void Solve()
{
  count = 0;
  double delta;
  double global_delta;
  double delta1, delta2;

  // Debug("Solve", 0);

  /* give global_delta a higher value then precision_goal */
  global_delta = 2 * precision_goal;
  if (proc_rank == 0)
  {
    errors = malloc(sizeof(double));
    errors[0] = global_delta;
    time_by_iteration = malloc(sizeof(double));
    time_by_iteration[0] = 0.0;
    time_by_iteration_size++;
  }
  while (global_delta > precision_goal && count < max_iter)
  {
    if (latency_flag)
    {
      latency = 0.0;
      byte = 0.0;
    }

    delta1 = Do_Step(0);
    Exchange_Borders();

    MPI_Barrier(grid_comm);

    delta2 = Do_Step(1);
    Exchange_Borders();

    delta = max(delta1, delta2);

    if (count % sweep == 0)
    {
      MPI_Allreduce(&delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, grid_comm);
    }

    count++;

    if (proc_rank == 0)
    {
      stop_timer();
      errors = realloc(errors, (count + 1) * sizeof(double));
      errors[count] = global_delta;
      time_by_iteration = realloc(time_by_iteration, (count + 1) * sizeof(double));
      time_by_iteration[count] = wtime;
      time_by_iteration_size++;
      resume_timer();
    }

    if (latency_flag)
    {
      latencies[count - 1] = latency;
      bytes[count - 1] = byte;
      if ((latencies = realloc(latencies, (count + 1) * sizeof(double))) == NULL)
        Debug("Solve : realloc(latencies) failed", 1);
      if ((bytes = realloc(bytes, (count + 1) * sizeof(double))) == NULL)
        Debug("Solve : realloc(bytes) failed", 1);
      latency_length = count;
    }
  }

  printf("(%i) Gridsize: %i,  Omega: %.2f, Iterations: %i, Error: %.2e\n", proc_rank, gridsize[X_DIR], omega, count, global_delta);
  current_iter = count;
}

void Write_Grid()
{
  int x, y;
  FILE *f;
  double **out;

  if (proc_rank == 0)
  {
    if ((out = malloc(gridsize[X_DIR] * sizeof(*out))) == NULL)
      Debug("Write_Grid : malloc failed", 1);
    if ((out[0] = malloc(gridsize[X_DIR] * gridsize[Y_DIR] * sizeof(**out))) == NULL)
      Debug("Write_Grid : malloc failed", 1);
    for (x = 1; x < gridsize[X_DIR]; x++)
      out[x] = out[0] + x * gridsize[Y_DIR];
    for (x = 0; x < gridsize[X_DIR]; x++)
      for (y = 0; y < gridsize[Y_DIR]; y++)
        out[x][y] = 0.0;

    char fn[200];
    // sprintf(fn, fn_template, P_grid[X_DIR], P_grid[Y_DIR], gridsize[X_DIR], gridsize[Y_DIR], omega);
    generate_fn(fn, "output", "phi");
    if ((f = fopen(fn, "w")) == NULL)
      Debug("Write_Grid : fopen failed", 1);

    // gather all data from other processes
    for (int i = 0; i < P; i++)
    {
      if (i > 0)
      {
        MPI_Recv(&phi[0][0], dim[X_DIR] * dim[Y_DIR], MPI_DOUBLE, i, 0, grid_comm, &status);
        MPI_Recv(&offset, 2, MPI_INT, i, 0, grid_comm, &status);
      }
      for (x = 1; x < dim[X_DIR] - 1; x++)
        for (y = 1; y < dim[Y_DIR] - 1; y++)
          out[offset[X_DIR] + x][offset[Y_DIR] + y] = phi[x][y];
      // fprintf(f, "%i %i %f\n", offset[X_DIR] + x, offset[Y_DIR] + y, phi[x][y]);
    }

    if (fwrite(out[0], sizeof(double), gridsize[X_DIR] * gridsize[Y_DIR], f) != gridsize[X_DIR] * gridsize[Y_DIR])
    {
      Debug("File write error.", 1);
      exit(1);
    }

    fclose(f);
  }
  else
  {
    MPI_Send(&phi[0][0], dim[X_DIR] * dim[Y_DIR], MPI_DOUBLE, 0, 0, grid_comm);
    MPI_Send(&offset, 2, MPI_INT, 0, 0, grid_comm);
  }
}

void Benchmark()
{
  double ***benchmark; /* 3D array holding benchmark results shape 2 x #processors x #omegas*/
  int benchmark_size, i, j, p;
  // Debug("Benchmark", 0);

  benchmark_size = 2 * P * omega_length;
  if (proc_rank == 0)
  {
    // Allocate memory for the pointers to the rows
    benchmark = malloc(2 * sizeof(double **));
    if (benchmark == NULL)
      Debug("Benchmark : malloc(benchmark) failed", 1);
    if ((benchmark[0] = malloc(P * sizeof(double *))) == NULL)
      Debug("Benchmark : malloc(benchmark[0]) failed", 1);
    if ((benchmark[1] = malloc(P * sizeof(double *))) == NULL)
      Debug("Benchmark : malloc(benchmark[1]) failed", 1);
    for (i = 0; i < 2; i++)
    {
      for (p = 0; p < P; p++)
      {
        if ((benchmark[i][p] = malloc(omega_length * sizeof(double))) == NULL)
          Debug("Benchmark : malloc(benchmark[i][p]) failed", 1);
      }
    }

    // Assign values to the elements
    for (i = 0; i < 2; i++)
    {
      for (p = 0; p < P; p++)
      {
        for (j = 0; j < omega_length; j++)
        {
          benchmark[i][p][j] = 0.0;
        }
      }
    }

    for (j = 0; j < omega_length; j++)
    {
      benchmark[0][0][j] = wtimes[j];
      benchmark[1][0][j] = cpu_util[j];
    }
  }

  // gather times
  if (proc_rank != 0)
  {
    MPI_Send(wtimes, omega_length, MPI_DOUBLE, 0, 0, grid_comm);
  }
  else
  {
    for (p = 1; p < P; p++)
    {
      MPI_Recv(benchmark[0][p], omega_length, MPI_DOUBLE, p, 0, grid_comm, &status);
    }
  }
  // gather times using MPI_Gather
  // MPI_Gather(&wtimes[0], omega_length, MPI_DOUBLE, &benchmark[0][proc_rank][0], omega_length, MPI_DOUBLE, 0, grid_comm);
  MPI_Barrier(grid_comm);

  // gather cpu util
  if (proc_rank != 0)
  {
    MPI_Send(cpu_util, omega_length, MPI_DOUBLE, 0, 0, grid_comm);
  }
  else
  {
    for (p = 1; p < P; p++)
    {
      MPI_Recv(benchmark[1][p], omega_length, MPI_DOUBLE, p, 0, grid_comm, &status);
    }
  }
  // MPI_Gather(&cpu_util[0], omega_length, MPI_DOUBLE, &benchmark[1][proc_rank][0], omega_length, MPI_DOUBLE, 0, grid_comm);
  MPI_Barrier(grid_comm);

  if (proc_rank == 0)
  {
    // char fn_template[] = "ppoisson_times/procg=%ix%i__gs=%ix%i_wl=%3.2f_wh=%3.2f_nomega=%i_times.dat";
    char fn[200];
    generate_fn(fn, "ppoisson_times", "times");
    FILE *f = fopen(fn, "w");
    if (f == NULL)
      Debug("Error opening benchmark file", 1);

    for (i = 0; i < 2; i++)
    {
      for (p = 0; p < P; p++)
      {
        for (j = 0; j < omega_length; j++)
        {
          fwrite(&benchmark[i][p][j], sizeof(double), 1, f);
        }
      }
    }
    fclose(f);

    //  save omega values to file
    generate_fn(fn, "ppoisson_times", "omegas");
    FILE *f2 = fopen(fn, "w");
    if (f2 == NULL)
      Debug("Error opening benchmark file", 1);

    for (i = 0; i < omega_length; i++)
    {
      fwrite(&omegas[i], sizeof(double), 1, f2);
    }

    fclose(f2);

    generate_fn(fn, "ppoisson_times", "iters");
    FILE *f3 = fopen(fn, "w");
    if (f3 == NULL)
      Debug("Error opening benchmark file", 1);

    for (i = 0; i < omega_length; i++)
    {
      fwrite(&iters[i], sizeof(int), 1, f3);
    }

    fclose(f3);
  }
}

void Error_Analysis()
{
  // Debug("Error_Analysis", 0);
  // char fn_template[] = "error_analysis/procg=%ix%i__gs=%ix%i_omega=%3.2f.dat";
  char fn[200];
  generate_fn(fn, "error_analysis", "");
  // sprintf(fn, fn_template, P_grid[X_DIR], P_grid[Y_DIR], gridsize[X_DIR], gridsize[Y_DIR], omega);
  if (proc_rank == 0)
  {
    FILE *f = fopen(fn, "w");
    if (f == NULL)
      Debug("Error opening error file", 1);

    for (int i = 0; i < count; i++)
    {
      fwrite(&errors[i], sizeof(double), 1, f);
    }
    fclose(f);
  }
}

void Sweep_Analysis()
{
  int sweep_vs_omega_size = omega_length * sweep_length;
  // Debug("Sweep_Analysis", 0);
  if (proc_rank == 0)
  {
    char fn[200];
    generate_fn(fn, "sweep_analysis", "iters");
    FILE *f = fopen(fn, "w");
    if (f == NULL)
      Debug("Error opening sweep file", 1);

    for (int i = 0; i < sweep_length; i++)
    {
      for (int j = 0; j < omega_length; j++)
      {
        fwrite(&iters_sweep_vs_omega[i][j], sizeof(int), 1, f);
      }
    }
    fclose(f);

    generate_fn(fn, "sweep_analysis", "times");
    FILE *f1 = fopen(fn, "w");
    if (f1 == NULL)
      Debug("Error opening sweep file", 1);

    for (int i = 0; i < sweep_length; i++)
    {
      for (int j = 0; j < omega_length; j++)
      {
        fwrite(&times_sweep_vs_omega[i][j], sizeof(int), 1, f);
      }
    }
    fclose(f1);
  }
}

void Latency_Analysis()
{
  // Debug("Latency_Analysis", 0);
  double ***out; // holds latencies, bytes per process shape =  2 x P x latency_length
  int out_size, i, j, p;

  if (proc_rank == 0)
  {

    out_size = 2 * P * latency_length;
    // Allocate memory for the pointers to the rows
    if ((out = malloc(2 * sizeof(double **))) == NULL)
      Debug("Latency_Analysis : malloc(out) failed", 1);
    if ((out[0] = malloc(P * sizeof(double *))) == NULL)
      Debug("Latency_Analysis : malloc(out[0]) failed", 1);
    if ((out[1] = malloc(P * sizeof(double *))) == NULL)
      Debug("Latency_Analysis : malloc(out[1]) failed", 1);
    for (i = 0; i < 2; i++)
    {
      for (p = 0; p < P; p++)
      {
        if ((out[i][p] = malloc(latency_length * sizeof(double))) == NULL)
          Debug("Latency_Analysis : malloc(out[i][p]) failed", 1);
      }
    }

    for (j = 0; j < latency_length; j++)
    {
      out[0][0][j] = latencies[j];
      out[1][0][j] = bytes[j];
    }
  }

  MPI_Barrier(grid_comm);

  // gather latencies
  if (proc_rank != 0)
  {
    MPI_Send(latencies, latency_length, MPI_DOUBLE, 0, 0, grid_comm);
  }
  else
  {
    for (p = 1; p < P; p++)
    {
      MPI_Recv(&out[0][p][0], latency_length, MPI_DOUBLE, p, 0, grid_comm, &status);
    }
  }

  MPI_Barrier(grid_comm);

  // gather bytes
  if (proc_rank != 0)
  {
    MPI_Send(bytes, latency_length, MPI_DOUBLE, 0, 0, grid_comm);
  }
  else
  {
    for (p = 1; p < P; p++)
    {
      MPI_Recv(&out[1][p][0], latency_length, MPI_DOUBLE, p, 0, grid_comm, &status);
    }
  }

  MPI_Barrier(grid_comm);

  if (proc_rank == 0)
  {
    char fn[200];
    generate_fn(fn, "latency_analysis", "");
    FILE *f = fopen(fn, "w");
    if (f == NULL)
      Debug("Error opening latency file", 1);

    for (i = 0; i < 2; i++)
    {
      for (p = 0; p < P; p++)
      {
        for (j = 0; j < latency_length; j++)
        {
          fwrite(&out[i][p][j], sizeof(double), 1, f);
        }
      }
    }
    fclose(f);
    // free memory
    // for (i = 0; i < 2; i++)
    // {
    //   for (p = 0; p < P; p++)
    //   {
    //     free(out[i][p]);
    //   }
    //   free(out[i]);
    // }
  }
}

void timeVIteration()
{
  char fn[200];
  if (proc_rank == 0)
  {
    generate_fn(fn, "timeviters", "");
    FILE *f = fopen(fn, "w");
    if (f == NULL)
      Debug("Error opening timeviter file", 1);

    for (int i = 0; i < time_by_iteration_size; i++)
    {
      fwrite(&time_by_iteration[i], sizeof(double), 1, f);
    }
    fclose(f);
  }
}

void Clean_Up_Problemdata()
{
  // Debug("Clean_Up", 0);

  free(phi[0]);
  free(phi);
  free(source[0]);
  free(source);
  // if (latency_flag)
  // {
  //   free(latencies);
  //   free(bytes);
  // }
}

void Clean_Up_Metadata()
{
  // Debug("Clean_Up_Metadata", 0);

  free(omegas);
  free(gridsizes);
  free(iters);
  free(wtimes);
  free(cpu_util);
  free(sweeps);
  // for (int i = 0; i < sweep_length; i++)
  // {
  //   free(iters_sweep_vs_omega[i]);
  // }
  // free(iters_sweep_vs_omega);
  // if (latency_flag)
  // {
  //   free(latencies);
  //   free(bytes);
  // }
}

void Setup_MPI_Datatypes()
{
  // Debug("Setup_MPI_Datatypes", 0);

  /* Datatype for vertical data exchange (Y_DIR) */
  MPI_Type_vector(dim[X_DIR] - 2, 1, dim[Y_DIR],
                  MPI_DOUBLE, &border_type[Y_DIR]);
  MPI_Type_commit(&border_type[Y_DIR]);

  /* Datatype for horizontal data exchange (X_DIR) */
  MPI_Type_vector(dim[Y_DIR] - 2, 1, 1,
                  MPI_DOUBLE, &border_type[X_DIR]);
  MPI_Type_commit(&border_type[X_DIR]);
}

void Exchange_Borders()
{
  // Debug("Exchange_Borders", 0);
  double latency_start;

  if (latency_flag)
  {
    // top to bottom and bottom to top exchange
    MPI_Barrier(grid_comm);
    latency_start = MPI_Wtime();
    MPI_Sendrecv(&phi[1][1], 1, border_type[Y_DIR], proc_top, 0,
                 &phi[1][dim[Y_DIR] - 1], 1, border_type[Y_DIR], proc_bottom, 0, grid_comm, &status);
    if (proc_top > 0)
    {
      latency += MPI_Wtime() - latency_start;
      byte += 2 * (dim[Y_DIR] - 2) * sizeof(double);
    }
    latency_start = MPI_Wtime();
    MPI_Sendrecv(&phi[1][dim[Y_DIR] - 2], 1, border_type[Y_DIR], proc_bottom, 0,
                 &phi[1][0], 1, border_type[Y_DIR], proc_top, 0, grid_comm, &status); /*  all  traffic in direction "bottom"        */
    if (proc_bottom > 0)
    {
      latency += MPI_Wtime() - latency_start;
      byte += 2 * (dim[Y_DIR] - 2) * sizeof(double);
    }
    // left to right and right to left exchange
    MPI_Barrier(grid_comm);
    latency_start = MPI_Wtime();
    MPI_Sendrecv(&phi[1][1], 1, border_type[X_DIR], proc_left, 0,
                 &phi[dim[X_DIR] - 1][1], 1, border_type[X_DIR], proc_right, 0, grid_comm, &status); /* all traffic in direction "left" */
    if (proc_left > 0)
    {
      latency += MPI_Wtime() - latency_start;
      byte += 2 * (dim[X_DIR] - 2) * sizeof(double);
    }
    latency_start = MPI_Wtime();
    MPI_Sendrecv(&phi[dim[X_DIR] - 2][1], 1, border_type[X_DIR], proc_right, 0,
                 &phi[0][1], 1, border_type[X_DIR], proc_left, 0, grid_comm, &status); /* all traffic in the direction "right" */
    if (proc_right > 0)
    {
      latency += MPI_Wtime() - latency_start;
      byte += 2 * (dim[X_DIR] - 2) * sizeof(double);
    }
  }
  else
  {
    MPI_Sendrecv(&phi[1][1], 1, border_type[Y_DIR], proc_top, 0,
                 &phi[1][dim[Y_DIR] - 1], 1, border_type[Y_DIR], proc_bottom, 0, grid_comm, &status);
    MPI_Sendrecv(&phi[1][dim[Y_DIR] - 2], 1, border_type[Y_DIR], proc_bottom, 0,
                 &phi[1][0], 1, border_type[Y_DIR], proc_top, 0, grid_comm, &status);
    MPI_Sendrecv(&phi[1][1], 1, border_type[X_DIR], proc_left, 0,
                 &phi[dim[X_DIR] - 1][1], 1, border_type[X_DIR], proc_right, 0, grid_comm, &status);
    MPI_Sendrecv(&phi[dim[X_DIR] - 2][1], 1, border_type[X_DIR], proc_right, 0,
                 &phi[0][1], 1, border_type[X_DIR], proc_left, 0, grid_comm, &status);
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  Setup_Proc_Grid(argc, argv);

  Get_CLIs(argc, argv);

  iters = malloc(omega_length * sizeof(int));
  wtimes = malloc(omega_length * sizeof(double));
  cpu_util = malloc(omega_length * sizeof(double));
  iters_sweep_vs_omega = malloc(sweep_length * sizeof(int *));
  times_sweep_vs_omega = malloc(sweep_length * sizeof(double *));
  for (int i = 0; i < sweep_length; i++)
  {
    iters_sweep_vs_omega[i] = malloc(omega_length * sizeof(int));
    times_sweep_vs_omega[i] = malloc(omega_length * sizeof(double));
  }

  if (latency_flag)
  {
    latencies = malloc(sizeof(double));
    bytes = malloc(sizeof(double));
  }

  for (grid_size_idx = 0; grid_size_idx < grid_length; grid_size_idx++)
  {
    for (int j = 0; j < sweep_length; j++)
    {
      sweep = sweeps[j];
      for (int i = 0; i < omega_length; i++)
      {
        omega = omegas[i];

        Setup_Grid();

        Setup_MPI_Datatypes();

        start_timer();

        Solve();

        stop_timer();

        if (write_output_flag)
        {
          Write_Grid();
        }

        if (track_errors)
        {
          Error_Analysis();
        }

        if (latency_flag)
        {
          Latency_Analysis();
        }

        // benchmarking
        iters[i] = current_iter;
        wtimes[i] = wtime;
        cpu_util[i] = 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime;

        MPI_Barrier(grid_comm);

        Clean_Up_Problemdata();
      }

      MPI_Barrier(grid_comm);

      if (benchmark_flag == 1)
      {
        Benchmark();
      }

      if (sweep_length > 1)
      {
        MPI_Reduce(&wtime, &wtime_sum, 1, MPI_DOUBLE, MPI_SUM, 0, grid_comm);
        if (proc_rank == 0)
        {
          for (int k = 0; k < omega_length; k++)
          {
            iters_sweep_vs_omega[j][k] = iters[k];
            times_sweep_vs_omega[j][k] = wtime_sum;
          }
        }
      }
    }

    if (sweep_length > 1)
    {
      Sweep_Analysis();
    }

    if (timeviter_flag == 1)
    {
      timeVIteration();
    }
  }

  MPI_Finalize();

  return 0;
}
