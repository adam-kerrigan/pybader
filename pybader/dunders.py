"""This module contains the dunders describing the metadata of the package.

This file is designed to be read without importing anything from packages not
required to build this package. It contains metadata executed by the setup.py
script and imported by the pybader.__init__ it is not meant to be used by
anything or anyone else.
"""
from sys import platform
import os

__name__ = "pybader"
__version__ = "0.3.1"
__author__ = "Adam Kerrigan"
__email__ = "ak1014@york.ac.uk"
__url__ = "https://github.com/kerrigoon/pybader"
__desc__ = "Threaded implementation of grid-based Bader charge analysis."
__long_desc__ = """Grid-based Bader charge analysis based on methods presented
in W. Tang, E. Sanville, and G. Henkelman A grid-based Bader analysis algorithm
without lattice bias, J. Phys.: Condens. Matter 21, 084204 (2009). Methods have
been updated to allow threading over multiple threads.
"""
__config__ = (os.path.join(os.getenv('LOCALAPPDATA'), 'pybader', 'config.ini')
        if platform == 'win32' else
        os.path.expanduser(os.path.join('~', '.config', 'bader', 'config.ini')))
