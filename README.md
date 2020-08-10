# pybader (v0.3.12)

pybader is a threaded implementation of grid-based Bader charge analysis. It performs grid-based Bader charge analysis as presented in [W Tang et al 2009 J. Phys.: Condens. Matter 21 084204]. Methods have been updated to allow threading over multiple threads. It runs on POSIX systems, Mac OS, and Windows.

## Usage

Installation will create 2 executeables (bader, bader-read) that can be used to run and read the output of the Bader calculation. To view available flags run:

```sh
$ bader --help
$ bader-read --help
```

By default the program bader will output a file called bader.p which is a pickled Bader object. This file can be interpreted by the bader-read script. Human readable files can also be created in place of the bader.p file by supplying the flag --output dat to the main bader program. If you would like this to be the default behaviour (bader.p files contain the entire charge denisty and voxel to Bader volume and/or atom map so can be quite large) there is a config.ini file with sections for all the configurable variables. The location of this varies depending on platform but can be found by checking the pybader.\_\_config\_\_ variable. The current default settings are shown below
```ini
[DEFAULT]
method = neargrid
refine_method = neargrid
vacuum_tol = None
refine_mode = ('changed', 2)
bader_volume_tol = 0.001
export_mode = None
prefix = ''
output = pickle
threads = 1
fortran_format = 0
speed_flag = False
spin_flag = False
```

Check the examples folder and function doc strings for information on how to interface with the python module.

```python
import pybader
from pybader import (
    methods,
    refinement,
    thread_handlers,
    interface,
)
import pybader.io
from inspect import getmembers, ismodule

help(pybader)
help(methods)
help(refinement)
help(thread_handlers)
help(interface)

for name, module in getmemebers(pybader.io, ismodule):
    help(module)
```

## Requirements

pyBader requires a small number of open source projects to work properly, if using pip to install they should be installed automatically:

* [Numpy] - NumPy is the fundamental package for scientific computing with Python.
* [Numba] - A Just-In-Time Compiler for Numerical Functions in Python.
* [tqdm] - A Fast, Extensible Progress Bar for Python and CLI.
* [pandas] - A Library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

## Installation

It's recommended to create a virtual python environment for the installation using a manager of your choice, this is optional and the docuemented manager is [pyenv]:

```sh
$ pyenv virtualenv pybader
$ pyenv activate pybader
```
Then the package can be installed using one of the methods described below:

### Pip

Use the standard pip installer

```sh
$ pip install pybader
```

### From source

Clone the repositry and make sure setuptools is installed

```sh
$ git clone https://github.com/kerrigoon/pybader
$ python pybader/setup.py install
```

### Development

Want to contribute? Great!
Open your favorite Terminal and run these commands.

```sh
$ git clone https://github.com/kerrigoon/pybader
$ pip install -e pybader/
```

Fiddle away and submit pull requests.

## License

MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [Numpy]: <https://numpy.org/>
   [Numba]: <https://numba.pydata.org/>
   [tqdm]: <https://tqdm.github.io/>
   [pandas]: <https://pandas.pydata.org/>
   [pyenv]: <https://github.com/pyenv/pyenv-virtualenv/>
   [W Tang et al 2009 J. Phys.: Condens. Matter 21 084204]: <https://doi.org/10.1088/0953-8984/21/8/084204>

