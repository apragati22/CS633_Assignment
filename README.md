## Directory Structure

When you unzip the project archive, the directory structure will look like this:

```
/Group01
├── src.c                        The main source code
├── Group01.pdf                  The report 
├── plot.py                      The python script used for plotting
├── parallel_IO_jobscripts       Contains the job scripts used (for src.c)
├── parallel_IO_outputs          Contains the output files generated (by src.c)
├── parallel_IO_plots            Contains the plots generated (for src.c)
├── sequential_io                Contains the code,jobscripts & outputs for seq io 
└── README.md          
```

## Usage of python script

To use the `plot.py` script, ensure you have Python 3 installed on your system. The script takes a folder path as an argument, processes the data files in the specified folder, and generates plots.

### Command

```bash
python3 plot.py ./folder_path
```
