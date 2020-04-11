from setuptools import setup
import os

CLASSIFIERS = """\
    Development Status :: 5 - Production/Stable
    Intended Audience :: Science/Research
    License :: OSI Approved
    License :: OSI Approved :: MIT License
    Natural Language :: English
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.6
    Programming Language :: Python :: 3.7
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3 :: Only
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Chemistry
    Topic :: Scientific/Engineering :: Physics
    Operating System :: Microsoft :: Windows
    Operating System :: POSIX
    Operating System :: Unix
    Operating System :: MacOS
"""

# Read metadata
with open(os.path.join('pybader', 'dunders.py'), 'r') as f:
    exec(f.read())

# List requirements
requirements = [
    'tqdm',
    'numba',
    'numpy',
]

# Entry points for console scripts
console_scripts = [
    'bader = pybader.entry_points:bader',
    'bader-read = pybader.entry_points:bader_read',
]

# Entry points for first run calibration
pybader_ep = [
    'create_config = pybader.entry_points:config_writer',
    'cache_JIT = pybader.entry_points:JIT_caching',
]

# Check for config file and move it to a back up
if os.path.isfile(__config__):
    os.rename(__config__, __config__[:-3] + 'bak')

setup(
    name=__name__,
    version=__version__,
    author=__author__,
    author_email=__email__,
    description=__desc__,
    long_description=__long_desc__,
    url=__url__,
    license="MIT",
    keywords=["grid-based", "Bader", "charge", "analysis"],
    packages=['pybader', os.path.join('pybader', 'io')],
    install_requires=requirements,
    python_requires='>=3.6',
    platforms=['Linux', 'Unix', 'Mac OS-X', 'Windows'],
    entry_points={
        'console_scripts': console_scripts,
        'pybader_ep': pybader_ep,
    },
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
)
