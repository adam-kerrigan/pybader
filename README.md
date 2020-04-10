# pybader (v0.1.1)

pyBader is a threaded implementation of grid-based Bader charge analysis. It performs grid-based Bader charge analysis as presented in W. Tang, E. Sanville, and G. Henkelman A grid-based Bader analysis algorithm without lattice bias, J. Phys.: Condens. Matter 21, 084204 (2009). Methods have been updated to allow threading over multiple threads.

## Requirements

pyBader requires a small number of open source projects to work properly, if using pip to install they should be installed automatically:

* [Numpy] - NumPy is the fundamental package for scientific computing with Python.
* [Numba] - A Just-In-Time Compiler for Numerical Functions in Python.
* [tqdm] - A Fast, Extensible Progress Bar for Python and CLI.

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
   [pyenv]: <https://https://github.com/pyenv/pyenv-virtualenv/>
