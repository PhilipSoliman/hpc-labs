/*
 * MPI_Fempois.c
 * 2D Poisson equation solver with MPI and FEM
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#define DEBUG 0

#define TYPE_GHOST 1
#define TYPE_SOURCE 2

#define INPUT_FOLDER "input"
#define OUTPUT_FOLDER "output"
#define BENCHMARK_FOLDER "benchmark"

#define MAXCOL 20

typedef struct
{
  int type;
  double x, y;
} Vertex;

typedef int Element[3];

typedef struct
{
  int Ncol;
  int *col;
  double *val;
} Matrixrow;

/* global variables */
double precision_goal; /* precision_goal of solution */
int max_iter;          /* maximum number of iterations alowed */
int P;                 /* total number of processes */
int P_grid[2];         /* processgrid dimensions */
MPI_Comm grid_comm;    /* grid COMMUNICATOR */
MPI_Status status;
int N_vert_total = 0;
int grid_size[2];
int do_adapt = 0; /* flag for adaptive refinement */

/* benchmark related variables */
clock_t ticks;    /* number of systemticks */
double wtime;     /* wallclock time */
int timer_on = 0; /* is timer running? */
double arbitrary_time;
double computation_time;
double exchange_time;
double communication_time;
double idle_time;
double io_time;
double total_time;

/* local process related variables */
int proc_rank;           /* rank of current process */
int proc_coord[2];       /* coordinates of current procces in processgrid */
int N_neighb;            /* Number of neighbouring processes */
int *proc_neighb;        /* ranks of neighbouring processes */
MPI_Datatype *send_type; /* MPI Datatypes for sending */
MPI_Datatype *recv_type; /* MPI Datatypes for receiving */

/* local grid related variables */
Vertex *vert; /* vertices */
double *phi;  /* vertex values */
int N_vert;   /* number of vertices */
Matrixrow *A; /* matrix A */

/* residual error related variables */
double *errors;
int N_iters;

void Setup_Proc_Grid();
void Setup_Grid();
void Build_ElMatrix(Element el);
void Sort_MPI_Datatypes();
void Setup_MPI_Datatypes(FILE *f);
void Exchange_Borders(double *vect);
void Solve();
void Write_Grid();
void Benchmark();
void Error_Analysis();
void Clean_Up();
void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();
void generate_filename(char *fn, char *folder, char *type);

void start_timer()
{
  if (!timer_on)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    ticks = clock();
    wtime = MPI_Wtime();
    timer_on = 1;
    computation_time = 0.0;
    exchange_time = 0.0;
    communication_time = 0.0;
    io_time = 0.0;
    idle_time = 0.0;
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
    printf("(%i) Elapsed Wtime: %14.6f s (%5.1f%% CPU)\n",
           proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
    resume_timer();
  }
  else
    printf("(%i) Elapsed Wtime:   %1.6f s (%5.1f%% CPU)\n",
           proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
}

void Debug(char *mesg, int terminate)
{
  if (DEBUG || terminate)
    printf("(%i) %s\n", proc_rank, mesg);
  if (terminate)
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
}

void generate_filename(char *fn, char *folder, char *type)
{
  sprintf(fn, "%s/nproc=%i_procg=%ix%i_grid=%ix%i_nvert=%i_adapt=%i_%s.dat",
          folder, P_grid[0]*P_grid[1], P_grid[0], P_grid[1], grid_size[0], grid_size[1],
          N_vert_total, do_adapt, type);
}

void Setup_Proc_Grid()
{
  FILE *f = NULL;
  char filename[50];
  int i;
  int N_nodes = 0, N_edges = 0;
  int *index, *edges, reorder;

  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  Debug("My_MPI_Init", 0);

  /* Retrieve the number of processes and current process rank */
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  /* Create process topology (Graph) */
  if (proc_rank == 0)
  {
    arbitrary_time = MPI_Wtime();
    sprintf(filename, "%s/mapping%i.dat", INPUT_FOLDER, P);
    if ((f = fopen(filename, "r")) == NULL)
      Debug("My_MPI_Init : Can't open mapping inputfile", 1);

    /* after reading N_nodes, a line is skipped */
    fscanf(f, "N_proc : %i\n%*[^\n]\n", &N_nodes);
    if (N_nodes != P)
      Debug("My_MPI_Init : Mismatch of number of processes in mapping inputfile", 1);
    io_time += MPI_Wtime() - arbitrary_time;
  }
  else
    N_nodes = P;

  if ((index = malloc(N_nodes * sizeof(int))) == NULL)
    Debug("My_MPI_Init : malloc(index) failed", 1);

  if (proc_rank == 0)
  {
    arbitrary_time = MPI_Wtime();
    for (i = 0; i < N_nodes; i++)
      fscanf(f, "%i\n", &index[i]);
    io_time += MPI_Wtime() - arbitrary_time;
  }

  arbitrary_time = MPI_Wtime();
  MPI_Bcast(index, N_nodes, MPI_INT, 0, MPI_COMM_WORLD);
  communication_time += MPI_Wtime() - arbitrary_time;

  N_edges = index[N_nodes - 1];
  if (N_edges > 0)
  {
    if ((edges = malloc(N_edges * sizeof(int))) == NULL)
      Debug("My_MPI_Init : malloc(edges) failed", 1);
  }
  else
    edges = index; /* this is actually nonsense,
                      but 'edges' needs to be a non-null pointer */

  if (proc_rank == 0)
  {
    arbitrary_time = MPI_Wtime();
    fscanf(f, "%*[^\n]\n"); /* skip a line of the file */
    for (i = 0; i < N_edges; i++)
      fscanf(f, "%i\n", &edges[i]);

    fclose(f);
    io_time += MPI_Wtime() - arbitrary_time;
  }

  arbitrary_time = MPI_Wtime();
  MPI_Bcast(edges, N_edges, MPI_INT, 0, MPI_COMM_WORLD);
  communication_time += MPI_Wtime() - arbitrary_time;

  reorder = 1;
  MPI_Graph_create(MPI_COMM_WORLD, N_nodes, index, edges, reorder, &grid_comm);

  /* Retrieve new rank of this process */
  MPI_Comm_rank(grid_comm, &proc_rank);

  if (N_edges > 0)
    free(edges);
  free(index);
}

void Setup_Grid()
{
  int i, j, v;
  Element element;
  int N_elm;
  char filename[50];
  FILE *f;

  Debug("Setup_Grid", 0);

  /* read general parameters (precision/max_iter) */
  if (proc_rank == 0)
  {
    arbitrary_time = MPI_Wtime();
    sprintf(filename, "%s/input.dat", INPUT_FOLDER);
    if ((f = fopen(filename, "r")) == NULL)
      Debug("Setup_Grid : Can't open input.dat", 1);
    fscanf(f, "precision goal: %lf\n", &precision_goal);
    fscanf(f, "max iterations: %i", &max_iter);
    fclose(f);

    sprintf(filename, "%s/gridsize.dat", INPUT_FOLDER);
    if ((f = fopen(filename, "r")) == NULL)
      Debug("Setup_Grid : Can't open gridsize.dat", 1);
    fscanf(f, "gridsize: %ix%i\n", &grid_size[0], &grid_size[1]);
    fscanf(f, "P_grid: %ix%i\n", &P_grid[0], &P_grid[1]);
    fscanf(f, "adapt: %i", &do_adapt);
    fclose(f);
    io_time += MPI_Wtime() - arbitrary_time;
  }

  arbitrary_time = MPI_Wtime();
  MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, grid_comm);
  MPI_Bcast(&max_iter, 1, MPI_INT, 0, grid_comm);
  communication_time += MPI_Wtime() - arbitrary_time;

  /* read process specific data */
  arbitrary_time = MPI_Wtime();
  sprintf(filename, "%s/input%i-%i.dat", INPUT_FOLDER, P, proc_rank);
  if ((f = fopen(filename, "r")) == NULL)
    Debug("Setup_Grid : Can't open data inputfile", 1);
  fscanf(f, "N_vert: %i\n%*[^\n]\n", &N_vert);
  io_time += MPI_Wtime() - arbitrary_time;

  /* allocate memory for phi and A */
  if ((vert = malloc(N_vert * sizeof(Vertex))) == NULL)
    Debug("Setup_Grid : malloc(vert) failed", 1);
  if ((phi = malloc(N_vert * sizeof(double))) == NULL)
    Debug("Setup_Grid : malloc(phi) failed", 1);

  if ((A = malloc(N_vert * sizeof(*A))) == NULL)
    Debug("Setup_Grid : malloc(*A) failed", 1);
  for (i = 0; i < N_vert; i++)
  {
    if ((A[i].col = malloc(MAXCOL * sizeof(int))) == NULL)
      Debug("Setup_Grid : malloc(A.col) failed", 1);
    if ((A[i].val = malloc(MAXCOL * sizeof(double))) == NULL)
      Debug("Setup_Grid : malloc(A.val) failed", 1);
  }

  /* init matrix rows of A */
  for (i = 0; i < N_vert; i++)
    A[i].Ncol = 0;

  /* Read all values */
  arbitrary_time = MPI_Wtime();
  for (i = 0; i < N_vert; i++)
  {
    fscanf(f, "%i", &v);
    fscanf(f, "%lf %lf %i %lf\n", &vert[v].x, &vert[v].y,
           &vert[v].type, &phi[v]);
  }
  io_time += MPI_Wtime() - arbitrary_time;

  /* build matrix from elements */
  arbitrary_time = MPI_Wtime();
  fscanf(f, "N_elm: %i\n%*[^\n]\n", &N_elm);
  io_time += MPI_Wtime() - arbitrary_time;
  for (i = 0; i < N_elm; i++)
  {
    arbitrary_time = MPI_Wtime();
    fscanf(f, "%*i"); /* we are not interested in the element-id */
    io_time += MPI_Wtime() - arbitrary_time;
    for (j = 0; j < 3; j++)
    {
      arbitrary_time = MPI_Wtime();
      fscanf(f, "%i", &v);
      io_time += MPI_Wtime() - arbitrary_time;
      element[j] = v;
    }
    arbitrary_time = MPI_Wtime();
    fscanf(f, "\n");
    io_time += MPI_Wtime() - arbitrary_time;
    arbitrary_time = MPI_Wtime();
    Build_ElMatrix(element);
    computation_time += MPI_Wtime() - arbitrary_time;
  }

  Setup_MPI_Datatypes(f);

  fclose(f);
}

void Add_To_Matrix(int i, int j, double a)
{
  int k;
  k = 0;

  while ((k < A[i].Ncol) && (A[i].col[k] != j))
    k++;
  if (k < A[i].Ncol)
    A[i].val[k] += a;
  else
  {
    if (A[i].Ncol >= MAXCOL)
      Debug("Add_To_Matrix : MAXCOL exceeded", 1);
    A[i].val[A[i].Ncol] = a;
    A[i].col[A[i].Ncol] = j;
    A[i].Ncol++;
  }
}

void Build_ElMatrix(Element el)
{
  int i, j;
  double e[3][2];
  double s[3][3];
  double det;

  e[0][0] = vert[el[1]].y - vert[el[2]].y; /* y1-y2 */
  e[1][0] = vert[el[2]].y - vert[el[0]].y; /* y2-y0 */
  e[2][0] = vert[el[0]].y - vert[el[1]].y; /* y0-y1 */
  e[0][1] = vert[el[2]].x - vert[el[1]].x; /* x2-x1 */
  e[1][1] = vert[el[0]].x - vert[el[2]].x; /* x0-x2 */
  e[2][1] = vert[el[1]].x - vert[el[0]].x; /* x1-x0 */

  det = e[2][0] * e[0][1] - e[2][1] * e[0][0];
  if (det == 0.0)
    Debug("One of the elements has a zero surface", 1);

  det = fabs(2 * det);

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      s[i][j] = (e[i][0] * e[j][0] + e[i][1] * e[j][1]) / det;

  for (i = 0; i < 3; i++)
    if (!((vert[el[i]].type & TYPE_GHOST) |
          (vert[el[i]].type & TYPE_SOURCE)))
      for (j = 0; j < 3; j++)
        Add_To_Matrix(el[i], el[j], s[i][j]);
}

void Sort_MPI_Datatypes()
{
  int i, j;
  MPI_Datatype data2;
  int proc2;

  for (i = 0; i < N_neighb - 1; i++)
    for (j = i + 1; j < N_neighb; j++)
      if (proc_neighb[j] < proc_neighb[i])
      {
        proc2 = proc_neighb[i];
        proc_neighb[i] = proc_neighb[j];
        proc_neighb[j] = proc2;
        data2 = send_type[i];
        send_type[i] = send_type[j];
        send_type[j] = data2;
        data2 = recv_type[i];
        recv_type[i] = recv_type[j];
        recv_type[j] = data2;
      }
}

void Setup_MPI_Datatypes(FILE *f)
{
  int i, s;
  int count;
  int *indices;
  int *blocklens;

  Debug("Setup_MPI_Datatypes", 0);

  arbitrary_time = MPI_Wtime();
  fscanf(f, "Neighbours: %i\n", &N_neighb);
  io_time += MPI_Wtime() - arbitrary_time;

  /* allocate memory */
  if (N_neighb > 0)
  {
    if ((proc_neighb = malloc(N_neighb * sizeof(int))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(proc_neighb) failed", 1);
    if ((send_type = malloc(N_neighb * sizeof(MPI_Datatype))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(send_type) failed", 1);
    if ((recv_type = malloc(N_neighb * sizeof(MPI_Datatype))) == NULL)
      Debug("Setup_MPI_Datatypes: malloc(recv_type) failed", 1);
  }
  else
  {
    proc_neighb = NULL;
    send_type = NULL;
    recv_type = NULL;
  }

  if ((indices = malloc(N_vert * sizeof(int))) == NULL)
    Debug("Setup_MPI_Datatypes: malloc(indices) failed", 1);
  if ((blocklens = malloc(N_vert * sizeof(int))) == NULL)
    Debug("Setup_MPI_Datatypes: malloc(blocklens) failed", 1);

  for (i = 0; i < N_vert; i++)
    blocklens[i] = 1;

  /* read vertices per neighbour */
  for (i = 0; i < N_neighb; i++)
  {
    arbitrary_time = MPI_Wtime();
    fscanf(f, "from %i :", &proc_neighb[i]);
    io_time += MPI_Wtime() - arbitrary_time;
    s = 1;
    count = 0;
    while (s == 1)
    {
      arbitrary_time = MPI_Wtime();
      s = fscanf(f, "%i", &indices[count]);
      io_time += MPI_Wtime() - arbitrary_time;

      if ((s == 1) && !(vert[indices[count]].type & TYPE_SOURCE))
      {
        count++;
      }
    }

    arbitrary_time = MPI_Wtime();
    fscanf(f, "\n");
    io_time += MPI_Wtime() - arbitrary_time;
    MPI_Type_indexed(count, blocklens, indices, MPI_DOUBLE, &recv_type[i]);
    MPI_Type_commit(&recv_type[i]);

    arbitrary_time = MPI_Wtime();
    fscanf(f, "to %i :", &proc_neighb[i]);
    io_time += MPI_Wtime() - arbitrary_time;
    s = 1;
    count = 0;
    while (s == 1)
    {
      arbitrary_time = MPI_Wtime();
      s = fscanf(f, "%i", &indices[count]);
      io_time += MPI_Wtime() - arbitrary_time;
      if ((s == 1) && !(vert[indices[count]].type & TYPE_SOURCE))
      {
        count++;
      }
    }
    arbitrary_time = MPI_Wtime();
    fscanf(f, "\n");
    io_time += MPI_Wtime() - arbitrary_time;
    MPI_Type_indexed(count, blocklens, indices, MPI_DOUBLE, &send_type[i]);
    MPI_Type_commit(&send_type[i]);
  }

  arbitrary_time = MPI_Wtime();
  Sort_MPI_Datatypes();
  computation_time += MPI_Wtime() - arbitrary_time;

  free(blocklens);
  free(indices);
}

void Exchange_Borders(double *vect)
{
  Debug("Exchange_Borders", 0);
  int i;

  if (N_neighb > 0)
  {
    for (i = 0; i < N_neighb; i++)
    {
      MPI_Sendrecv(&phi[0], 1, send_type[i], proc_neighb[i], 0,
                   &phi[0], 1, recv_type[i], proc_neighb[i], 0, grid_comm, &status);
    }
  }
}

void Solve()
{
  int count = 0;
  int i, j;
  double *r, *p, *q;
  double a, b, r1, r2 = 1;

  double sub;

  Debug("Solve", 0);

  if ((r = malloc(N_vert * sizeof(double))) == NULL)
    Debug("Solve : malloc(r) failed", 1);
  if ((p = malloc(N_vert * sizeof(double))) == NULL)
    Debug("Solve : malloc(p) failed", 1);
  if ((q = malloc(N_vert * sizeof(double))) == NULL)
    Debug("Solve : malloc(q) failed", 1);

  /* Implementation of the CG algorithm : */

  arbitrary_time = MPI_Wtime();
  Exchange_Borders(phi);
  exchange_time += MPI_Wtime() - arbitrary_time;

  /* r = b-Ax */
  arbitrary_time = MPI_Wtime();
  for (i = 0; i < N_vert; i++)
  {
    r[i] = 0.0;
    for (j = 0; j < A[i].Ncol; j++)
      r[i] -= A[i].val[j] * phi[A[i].col[j]];
  }

  r1 = 2 * precision_goal;
  if (proc_rank == 0)
  {
    if ((errors = malloc(sizeof(double))) == NULL)
        Debug("Solve : malloc(errors) failed", 1);
  }
  computation_time += MPI_Wtime() - arbitrary_time;
  while ((count < max_iter) && (r1 > precision_goal))
  {
    /* r1 = r' * r */
    arbitrary_time = MPI_Wtime();
    sub = 0.0;
    for (i = 0; i < N_vert; i++)
      if (!(vert[i].type & TYPE_GHOST))
        sub += r[i] * r[i];
    computation_time += MPI_Wtime() - arbitrary_time;

    arbitrary_time = MPI_Wtime();
    MPI_Allreduce(&sub, &r1, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
    communication_time += MPI_Wtime() - arbitrary_time;

    arbitrary_time = MPI_Wtime();
    if (count == 0)
    {
      /* p = r */
      for (i = 0; i < N_vert; i++)
        p[i] = r[i];
    }
    else
    {
      b = r1 / r2;

      /* p = r + b*p */
      for (i = 0; i < N_vert; i++)
        p[i] = r[i] + b * p[i];
    }
    computation_time += MPI_Wtime() - arbitrary_time;

    arbitrary_time = MPI_Wtime();
    Exchange_Borders(p);
    exchange_time += MPI_Wtime() - arbitrary_time;

    /* q = A * p */
    arbitrary_time = MPI_Wtime();
    for (i = 0; i < N_vert; i++)
    {
      q[i] = 0;
      for (j = 0; j < A[i].Ncol; j++)
        q[i] += A[i].val[j] * p[A[i].col[j]];
    }

    /* a = r1 / (p' * q) */
    sub = 0.0;
    for (i = 0; i < N_vert; i++)
      if (!(vert[i].type & TYPE_GHOST))
        sub += p[i] * q[i];
    computation_time += MPI_Wtime() - arbitrary_time;
    arbitrary_time = MPI_Wtime();
    MPI_Allreduce(&sub, &a, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
    communication_time += MPI_Wtime() - arbitrary_time;

    arbitrary_time = MPI_Wtime();
    a = r1 / a;

    /* x = x + a*p */
    for (i = 0; i < N_vert; i++)
      phi[i] += a * p[i];

    /* r = r - a*q */
    for (i = 0; i < N_vert; i++)
      r[i] -= a * q[i];
    computation_time += MPI_Wtime() - arbitrary_time;

    r2 = r1;

    if (proc_rank == 0)
    {
      errors[count] = r1;
      if ((errors = realloc(errors, (count + 2) * sizeof(double))) == NULL)
        Debug("Solve : realloc(errors) failed", 1);
    }
    count++;
  }
  free(q);
  free(p);
  free(r);

  if (proc_rank == 0)
  {
    printf("Number of iterations : %i\n", count);
    N_iters = count;
  }
}

void Write_Grid()
{
  int i, j;
  char filename[100];
  FILE *f;
  double **out;
  double *tmp;
  int out_size = N_vert;

  Debug("Write_Grid", 0);

  if ((out = malloc(N_vert * sizeof(double *))) == NULL)
    Debug("Write_Grid : malloc(out) failed", 1);
  for (i = 0; i < N_vert; i++)
    if ((out[i] = malloc(3 * sizeof(double))) == NULL)
      Debug("Write_Grid : malloc(out[i]) failed", 1);

  for (i = 0; i < N_vert; i++)
  {
    if (vert[i].type & TYPE_GHOST)
    {
      out[i][0] = 0.0;
      out[i][1] = 0.0;
      out[i][2] = 0.0;
    }
    else
    {
      out[i][0] = vert[i].x;
      out[i][1] = vert[i].y;
      out[i][2] = phi[i];
    }
  }

  arbitrary_time = MPI_Wtime();
  sprintf(filename, "%s/nproc=%i_proc=%i.dat", OUTPUT_FOLDER, P, proc_rank);
  if ((f = fopen(filename, "w")) == NULL)
    Debug("Write_Grid : Can't open data outputfile", 1);

  if (fwrite(out, sizeof(double), N_vert * 3, f) != 3 * N_vert)
    Debug("Write_Grid : Error during writing", 1);

  fclose(f);
  io_time += MPI_Wtime() - arbitrary_time;

  // collect sizes of output files
  int *sizes;
  if (proc_rank == 0)
  {
    if ((sizes = malloc(P * sizeof(int))) == NULL)
      Debug("Write_Grid : malloc(sizes) failed", 1);
  }

  arbitrary_time = MPI_Wtime();
  MPI_Gather(&out_size, 1, MPI_INT, sizes, 1, MPI_INT, 0, grid_comm);
  communication_time += MPI_Wtime() - arbitrary_time;

  // combine output files into one on root process
  MPI_Barrier(grid_comm);
  if (proc_rank == 0)
  {
    int read_size;
    for (i = 0; i < P; i++)
      N_vert_total += sizes[i];
      printf("N_vert_total: %d\n", N_vert_total);

    // allocate memory for combined file
    if ((tmp = malloc(3 * N_vert_total * sizeof(double))) == NULL)
      Debug("Write_Grid : malloc(tmp) failed", 1);

    // gather output
    for (i = 0; i < P; i++)
    {
      out_size = sizes[i];

      arbitrary_time = MPI_Wtime();
      sprintf(filename, "%s/nproc=%i_proc=%i.dat", OUTPUT_FOLDER, P, i);
      printf("filename: %s\n", filename);
      if ((f = fopen(filename, "r")) == NULL)
        Debug("Write_Grid : Can't open data outputfile", 1);
      io_time += MPI_Wtime() - arbitrary_time;

      read_size = fread(&tmp[3 * (out_size - sizes[0])], sizeof(double), 3 * out_size, f);
      // printf("read_size: %d\n", read_size);
      // printf("out_size: %d\n", out_size);
      if (read_size != 3 * out_size)
      {
        Debug("Write_Grid : Error during reading", 1);
      }

      arbitrary_time = MPI_Wtime();
      fclose(f);
      io_time += MPI_Wtime() - arbitrary_time;

      // delete file
      remove(filename);
    }

    FILE *f_combined;
    generate_filename(filename, OUTPUT_FOLDER, "combined");
    arbitrary_time = MPI_Wtime();
    if ((f_combined = fopen(filename, "w")) == NULL)
      Debug("Write_Grid : Can't open combined data outputfile", 1);

    if (fwrite(tmp, sizeof(double), 3 * N_vert_total, f_combined) != 3 * N_vert_total)
      Debug("Write_Grid : Error during writing", 1);

    fclose(f_combined);
    io_time += MPI_Wtime() - arbitrary_time;
  }

  free(out);
  if (proc_rank == 0)
    free(sizes);

  MPI_Barrier(grid_comm);
}

void Benchmark()
{
  stop_timer();
  idle_time = wtime - computation_time - communication_time - io_time - exchange_time;
  total_time = computation_time + communication_time + idle_time + io_time + exchange_time;
  printf("(%i) Computation time:    %1.6f (%4.2f\%)\n", proc_rank, computation_time, 100.0 * computation_time / total_time);
  printf("(%i) Exchange time:       %1.6f (%4.2f\%)\n", proc_rank, exchange_time, 100.0 * exchange_time / total_time);
  printf("(%i) Communication time:  %1.6f (%4.2f\%)\n", proc_rank, communication_time, 100.0 * communication_time / total_time);
  printf("(%i) Idle time:           %1.6f (%4.2f\%)\n", proc_rank, idle_time, 100.0 * idle_time / total_time);
  printf("(%i) I/O time:            %1.6f (%4.2f\%)\n", proc_rank, io_time, 100.0 * io_time / total_time);
  print_timer();

  // save all times to binary file as one array
  double **out;
  double *tmp;
  int *displacement;
  int *sizes;

  if ((sizes = malloc(P * sizeof(int))) == NULL)
    Debug("Benchmark : malloc(sizes) failed", 1);
  if ((displacement = malloc(P * sizeof(int))) == NULL)
    Debug("Benchmark : malloc(displacement) failed", 1);

  for (int i = 0; i < P; i++)
  {
    sizes[i] = 5;
    displacement[i] = i * 5;
  }

  if ((tmp = malloc(5 * sizeof(double))) == NULL)
    Debug("Benchmark : malloc(tmp) failed", 1);

  tmp[0] = computation_time;
  tmp[1] = exchange_time;
  tmp[2] = communication_time;
  tmp[3] = idle_time;
  tmp[4] = io_time;

  if ((out = malloc(P * sizeof(double *))) == NULL)
    Debug("Benchmark : malloc(out) failed", 1);
  for (int i = 0; i < P; i++)
    if ((out[i] = malloc(5 * sizeof(double))) == NULL)
      Debug("Benchmark : malloc(out[i]) failed", 1);

  // collect all times into out array
  MPI_Gatherv(tmp, 5, MPI_DOUBLE, out, sizes, displacement, MPI_DOUBLE, 0, grid_comm);

  if (proc_rank == 0)
  {
    FILE *f;
    char filename[100];
    generate_filename(filename, BENCHMARK_FOLDER, "times");
    if ((f = fopen(filename, "w")) == NULL)
      Debug("Benchmark : Can't open times outputfile", 1);

    if (fwrite(out, sizeof(double), 5 * P, f) != 5 * P)
      Debug("Benchmark : Error during writing", 1);

    fclose(f);
  }
}

void Error_Analysis()
{
  if (proc_rank == 0)
  {
    FILE *f;
    char filename[100];
    generate_filename(filename, BENCHMARK_FOLDER, "error");
    if ((f = fopen(filename, "w")) == NULL)
      Debug("Error_Analysis : Can't open error outputfile", 1);

    if (fwrite(errors, sizeof(double), N_iters, f) != N_iters)
      Debug("Error_Analysis : Error during writing", 1);

    fclose(f);
  }
}

void Clean_Up()
{
  int i;
  Debug("Clean_Up", 0);

  if (N_neighb > 0)
  {
    free(recv_type);
    free(send_type);
    free(proc_neighb);
  }

  for (i = 0; i < N_vert; i++)
  {
    free(A[i].col);
    free(A[i].val);
  }
  free(A);
  free(vert);
  free(phi);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  Setup_Proc_Grid();

  Setup_Grid();

  start_timer();

  Solve();

  Write_Grid();

  Benchmark();

  Error_Analysis();

  Clean_Up();

  Debug("MPI_Finalize", 0);

  MPI_Finalize();

  return 0;
}
