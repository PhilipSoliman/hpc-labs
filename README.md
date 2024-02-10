# hpc-labs
This repo
- contains all code of hpc lab assignments + my answers;
- Additionally it processes output data from HPC labs
using Python's visualisation libraries; and
- contains my report.

The rest of this README is the original README of the HPC labs.

# Assignment HPC

This file contains a bunch of helper commands and links that you can find useful during your development on DelftBlue.

Please unzip the assignments folder and then add `<assignment_x>` folder from `HPC/<assignment_x>` in the `HPC` folder of your delftBlue. The assignments folders can be downloaded from BrightSpace.

Each assignment folders contains .sh file, this are files that contains slurm directive and tell DelftBlue how to compile and run your code. You can submit a job to the cluster by doing `sbatch ./HPC/<assigment_folder>/<program_name>.sh`

To change the directive for slurm, you need to modify the `<program_name>.sh` files. 
For example to run the code on 4 nodes `SBATCH --ntasks=4`.
Please look at [slurm directive docs](https://slurm.schedmd.com/sbatch.html) for a details list of all options and their corresponding syntax

# Python analyses 
Each assignment directory has its own analysis.py file:
- [assignment 0](intro_assignment/analysis.py);
- [assignment 1](assignment_1/analysis.py);
- [assignment 2](assignment_2/analysis.py);
- [assignment 3](assignment_3/analysis.py).

These scripts contains the code to process the output data from the HPC labs using Python's visualisation libraries. In each of these scripts some custom modules are used, which are located in the [python_utils][python_utils] directory. For python to be able to import these modules, the [python_utils][python_utils] directory should be added to the PYTHONPATH environment variable. This can be done on VSCode by adding the following line to the `.vscode/settings.json` file in the workspace directory:
```json
{
    "python.environmentVariables": {
        "PYTHONPATH": "${workspaceFolder}/python_utils"
    }
}
```
Alternatively, the PYTHONPATH environment variable can be set by on windows by adding the following line to the `.vscode/settings.json` file in the workspace directory:
```json
{
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "${env:PYTHONPATH};${workspaceFolder}/python_utils",
  }
}
```
Yet another way is by running the following command in the terminal:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/python_utils
```
where `/path/to/python_utils` should be replaced with the path to the [python_utils][python_utils] directory.


# Resources

[DelftBlue docs](https://doc.dhpc.tudelft.nl/delftblue/)

Connect to DelftBlue
`ssh <netid>@login.delftblue.tudelft.nl`1

Schedule the job (return job id) 
`sbatch ./HPC/<assigment_folder>/<main>.sh`

Check your queue of job 
`squeue -u <netid>`

Show details about your job (really useful) 
`scontrol show job <jobid>`

To cancel job 
`scancel <jobid>`

View jobs 
`sacct`

[OpenDemand Platform](https://opendemand.dhpc.tudelft.nl/)

[DefltBlue dashboard for cluster status](https://login.delftblue.tudelft.nl/pun/sys/dashboard)

Sbatch directive to use the HPC course credits to submit your job (to add to .sh files)
`#SBATCH –-account=Education-EEMCS-Courses-IN4049TU`

Storage node (not accesible on compute node)
`/tudelft.net/` 

Transfer file to DelftBlue manually
`rsync -av ../HPC <netid>@login.delftblue.tudelft.nl:~/`
