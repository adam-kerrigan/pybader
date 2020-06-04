"""This module is not intentded to be accessed.

Module serves as entry points for the installed bader and bader read scripts as
well as the first run install. It is not designed to accessed as a module as
most of the functions within require sys.argv to be set or often don't complete
correctly if used out of turn.
"""
import os
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
from inspect import getmembers, isfunction, ismodule
from pickle import dump, load
from time import time

import numpy as np

from pybader import __config__
from pybader import __doc__ as doc
from pybader import __version__, io, methods, refinement, utils
from pybader.interface import Bader, python_config
from pybader.jits import jit_functs
from pybader.utils import nostdout, tqdm_wrap


def bader():
    splash = f"""\n  Bader Charge Analysis ({__version__})
    """

    # source avail configs
    config = ConfigParser()
    config.read(__config__)

    description = doc
    filename = """Path to file containing a density,"""
    b = """Method to use for Bader partitioning:
    ongrid | neargrid (default)"""
    r = """How to refine points, check for all edge points or only updated
    points. Optional inclusion of an integer to specify how many interations to
    do, if not supplied iterate until no points change:
    all | changed (default) [int]"""
    ref = """Path to file containing a reference density, if two files are
    supplied the files will be read and then summed"""
    vac = """Tolerance used for assuming density is vacuum:
    auto (1E-3) | float"""
    e = """Bader volumes or atoms to be exported. Optional string to select
    mode, 'sel' modes require at least one integer to be supplied indicating
    volume or atom index:
    [ sel_atom (default) | sel_volume | all_atom | all_volume ] int [int ...]"""
    p = """Prefix to apply to the outputted file, defaults to none.
    Supplying nothing after this flag will set the output to the same directory
    as the file"""
    i = """File-type of the filename argument, if not supplied a guess will be
    made from whether known extensions are in the filename"""
    j = """How many threads to spread the calculation over. Note: actual threads
    used will be +1 to the value supplied here as progress bars operate in
    seperate thread. To remove progress bars and run on a single thread use -j 0
    """
    s = """Read in the spin density and print out the spin associted with each
    Bader atom. Only appicable if file contains both charge and spin densities
    """
    x = """Perform refinement of volumes after they have been assigned to atoms
    and only refine atom boundaries"""
    f = """Whether to use increase the 'fortran-likeness' of the output file.
    This flag is a counter with max value of 2. 0 (default) exports in python
    format to the precsion of the input file, this means column width is not
    constant if negative numbers are involved. 1 exports again in python format
    but with a space infront of positive numbers to maintain column width, this
    comes with a mild overhead. 2 puts into fortran 'standard' form and replaces
    the leading zero with a minus for negative numbers in output files, this
    adds an overhead to the write-out process but matches the output of some
    fortran programs"""
    o = """How to save the information. Pickle (default) the entire class or just
    print a text file containing the information about the Bader atoms and
    volumes"""
    c = f"""Load a profile from the config file located at '{__config__}'
    """

    bchoice = methods.__contains__
    i_gen = (name for name, module in getmembers(io, ismodule)
             if module.__extensions__ is not None)
    ichoice = [name for name in i_gen]
    rchoice = ['all', 'changed']
    ochoice = ['pickle', 'dat']
    cchoice = config.keys()
    export_check = ['all_atoms', 'all_volumes', 'sel_atoms', 'sel_volumes']

    # arguements of the program
    parser = ArgumentParser(description=description)
    parser.add_argument('filename', nargs=1, help=filename)
    parser.add_argument('-m', '--method', nargs=1, choices=bchoice, help=b)
    parser.add_argument('-r', '--refine', nargs='+', help=r)
    parser.add_argument('-ref', '--reference', nargs='+', help=ref)
    parser.add_argument('-vac', '--vacuum-tol', nargs=1, help=vac)
    parser.add_argument('-e', '--export', nargs='+', help=e)
    parser.add_argument('-p', '--prefix', nargs='?', const=False, help=p)
    parser.add_argument('-i', '--file-type', nargs=1, choices=ichoice, help=i)
    parser.add_argument('-j', '--threads', nargs=1, type=int, help=j)
    parser.add_argument('-s', '--spin', action='store_true', help=s)
    parser.add_argument('-x', '--speed', action='store_true', help=x)
    parser.add_argument('-f', '--fortran-format', action='count', help=f)
    parser.add_argument('-o', '--output', nargs=1, choices=ochoice, help=o)
    parser.add_argument('-c', '--config', nargs=1, choices=cchoice, help=c)

    # convert args to dict and load default config
    args = vars(parser.parse_args())
    config_key = args['config'][0] if args['config'] is not None else 'DEFAULT'
    config = python_config(__config__, config_key)

    print(splash)

    # port args to config
    if args.get('method') is not None:
        config['method'] = args['method'][0]
        config['refine_method'] = config['method']
    if args.get('refine') is not None:
        try:
            iters = int(args['refine'][0])
            mode = 'changed'
        except ValueError:
            if args['refine'][0] in rchoice:
                mode = args['refine'][0]
            else:
                mode = 'changed'
                print("  Unable to parse refinement mode, using changed\n")
            if len(args['refine']) == 2:
                iters = int(args['refine'][1])
            else:
                iters = -1
        config['refine_mode'] = (mode, iters)
    if args.get('vacuum_tol') is not None:
        try:
            config['vacuum_tol'] = np.float64(args['vacuum_tol'][0])
        except ValueError:
            if args['vacuum_tol'][0].lower() != 'auto':
                print("  Unable to parse vacuum tolerance, using 1E-3\n")
            config['vacuum_tol'] = 1E-3
    if args.get('export') is not None:
        try:
            export_list = np.array(args['export'], dtype=np.int64)
            export_type = 'atoms'
        except ValueError:
            if len(args['export']) == 1:
                export_list = [-2]
                if args['export'][0] in export_check:
                    export_type = args['export'][0][4:]
                else:
                    print("  Unable to parse export type, using all_atoms\n")
                    export_type = 'atoms'
            else:
                export_list = np.array(args['export'][1:], dtype=np.int64)
                if args['export'][0] in export_check:
                    export_type = args['export'][0].split('_')[-1]
                else:
                    print("  Unable to parse export type, using sel_atoms\n")
                    export_type = 'atoms'
        finally:
            config['export_mode'] = (export_type, export_list)
    if args.get('file_type') is not None:
        config['file_type'] = args['file_type'][0]
    if args.get('threads') is not None:
        config['threads'] = args['threads'][0]
    if args.get('spin'):
        config['spin_flag'] = not config['spin_flag']
    if args.get('speed'):
        config['speed_flag'] = not config['speed_flag']
    if args.get('fortran_format') is not None:
        config['fortran_format'] += args['fortran_format']
        config['fortran_format'] %= 3
    if args.get('prefix') is not None and args.get('prefix'):
        config['prefix'] = args['prefix']
    if args.get('output') is not None:
        config['output'] = args['output'][0]

    # read filename and init bader class
    t0 = time()
    fname = args.get('filename')[0]
    bader = Bader.from_file(fname, **config)

    # check if prefix should be relative to filename
    if args.get('prefix') is not None and not args.get('prefix'):
        bader.prefix = bader.info['prefix']

    # read in reference densities
    if args.get('reference') is not None:
        ftype = config.get('file_type', None)
        bader.reference = np.zeros(bader.density.shape, dtype=np.float64)
        for ref in args['reference']:
            ref_den = Bader.from_file(ref, file_type=ftype).charge
            try:
                bader.reference[:] = ref_den
            except ValueError:
                print("  ERROR: Reference and density have different grids.")
                exit()
    bader()
    print(f"\n  Total time taken {time() - t0:.3f}s\n")


def bader_read():
    description = "Tool for viewing the output of the bader program"
    filename = """Path to file containing Bader output, if no path is supplied a
    default value of './bader.p' is used"""
    a = "Whether to show the Bader atom infomation"
    v = "Whether to show the Bader volume infomation"
    e = """Bader volumes or atoms to be exported:
    [ sel_atom (default) | sel_volume | all_atom | all_volume ] int [int ...]"""
    d = "Write copy of the orginal density file"
    f = """Whether to use increase the 'fortran-likeness' of the output file.
    This flag is a counter with max value of 2. 0 (default) exports in python
    format to the precsion of the input file, this means column width is not
    constant if negative numbers are involved. 1 exports again in python format
    but with a space infront of positive numbers to maintain column width, this
    comes with a mild overhead. 2 puts into fortran 'standard' form and replaces
    the leading zero with a minus for negative numbers in output files, this
    adds an overhead to the write-out process but matches the output of some
    fortran programs"""
    r = "Recast pickled class as new class"

    export_check = ['all_atoms', 'all_volumes', 'sel_atoms', 'sel_volumes']

    parser = ArgumentParser(description=description)
    parser.add_argument('filename', nargs='?',
                        default='bader.p', help=filename)
    parser.add_argument('-a', '--atoms', action='store_true', help=a)
    parser.add_argument('-v', '--volume', action='store_true', help=v)
    parser.add_argument('-e', '--export', nargs='+', help=e)
    parser.add_argument('-d', '--density-write', action='store_true', help=d)
    parser.add_argument('-f', '--fortran-format', action='count', help=f)
    parser.add_argument('-r', '--recast', action='store_true', help=r)
    args = vars(parser.parse_args())

    with open(args['filename'], '+rb') as f:
        bader = load(f)

    if args['fortran_format'] is not None:
        bader.fortran_format = args['fortran_format'] % 3
    if args.get('export') is not None:
        try:
            export = np.array(args['export'], dtype=np.int64)
            export_type = 'atoms'
        except ValueError:
            if len(args['export']) == 1:
                export = [-2]
                if args['export'][0] in export_check:
                    export_type = args['export'][0][4:]
                else:
                    print("  Unable to parse export type, using all_atoms\n")
                    export_type = 'atoms'
            else:
                export = np.array(args['export'][1:], dtype=np.int64)
                if args['export'][0] in export_check:
                    export_type = args['export'][0].split('_')[-1]
                else:
                    print("  Unable to parse export type, using sel_atoms\n")
                    export_type = 'atoms'
        finally:
            bader.export_mode = (export_type, export)
            bader.prefix = ''
        print(f"  Writing Bader {export_type} to file:")
        if export_type == 'volumes':
            if export == -2:
                for vol_num in range(bader.bader_maxima.shape[0]):
                    bader.write_volume(vol_num)
                if bader.vacuum_tol is not None:
                    bader.write_volume(-1)
            else:
                for vol_num in export:
                    bader.write_volume(vol_num)
        elif export_type == 'atoms':
            if export == -2:
                for vol_num in range(bader.atoms.shape[0]):
                    bader.write_volume(vol_num)
                if bader.vacuum_tol is not None:
                    bader.write_volume(-1)
            else:
                for vol_num in export:
                    bader.write_volume(vol_num)
    if args['volume']:
        print(bader.results(volume_flag=True))
    if args['density_write']:
        bader.write_density()
    if args['atoms']:
        print(bader.results())
    if args['recast']:
        new_bader = Bader.from_dict(bader.as_dict)
        with open(filename, '+wb') as f:
            dump(new_bader, f)


def config_writer():
    old_config = None
    print(f"  Writing default config to '{__config__}': ", end='')
    if not os.path.exists(os.path.dirname(__config__)):
        os.makedirs(os.path.dirname(__config__))
    elif os.path.isfile(__config__[:-3] + 'bak'):
        os.rename(__config__[:-3] + 'bak', __config__)
        old_config = ConfigParser()
        with open(__config__, 'r') as f:
            old_config.read_file(f)

    config = ConfigParser()
    config['DEFAULT'] = {
        'method': 'neargrid',
        'refine_method': 'neargrid',
        'vacuum_tol': 'None',
        'refine_mode': ('changed', 2),
        'bader_volume_tol': 1E-3,
        'export_mode': 'None',
        'prefix': "''",
        'output': 'pickle',
        'threads': 1,
        'fortran_format': 0,
        'speed_flag': False,
        'spin_flag': False,
    }
    config['speed'] = {
        'method': 'ongrid',
        'refine_method': 'neargrid',
        'refine_mode': ('changed', 3),
        'speed_flag': True,
    }

    if old_config is not None:
        for key in old_config:
            if key not in config:
                config[key] = {}
            for keyword in old_config[key]:
                config[key][keyword] = old_config[key].get(keyword)
    with open(__config__, 'w') as f:
        config.write(f)
    print("Done.")


def JIT_caching():
    desc = "Caching JIT'd functions:"
    tot = 0
    for _, val in jit_functs['methods'].items():
        tot += len(val)
    for _, val in jit_functs['refinement'].items():
        tot += len(val)
    for _, val in jit_functs['utils'].items():
        tot += len(val)
    with tqdm_wrap(total=tot, desc=desc, file=sys.stderr) as pbar:
        for key, val in jit_functs['utils'].items():
            for args in val:
                getattr(utils, key)(*args)
                pbar.update(1)
        for key, val in jit_functs['methods'].items():
            for args in val:
                getattr(methods, key)(*args)
                pbar.update(1)
        for key, val in jit_functs['refinement'].items():
            for args in val:
                getattr(refinement, key)(*args)
                pbar.update(1)
