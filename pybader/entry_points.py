"""This module is not intentded to be accessed.

Module serves as entry points for the installed bader and bader read scripts as
well as the first run install. It is not designed to accessed as a module as
most of the functions within require sys.argv to be set or often don't complete
correctly if used out of turn.
"""
import numpy as np
import sys
import os
from pickle import (
        load,
        dump,
)
from argparse import ArgumentParser
from configparser import ConfigParser
from time import time
from pybader import (
        __version__,
        __doc__ as doc,
        __config__,
        methods,
        io,
)
from pybader.interface import (
        Bader,
        python_config,
)
from pybader.utils import (
        array_assign,
        array_merge,
        factor_3d,
        nostdout,
        vacuum_assign,
        volume_extend,
        volume_merge,
        volume_offset,
        tqdm_wrap,
)
from inspect import (
        getmembers,
        ismodule,
        isfunction,
)


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
    o="""How to save the information. Pickle (default) the entire class or just
    print a text file containing the information about the Bader atoms and
    volumes"""
    c=f"""Load a profile from the config file located at '{__config__}'
    """

    bchoice = methods.__contains__
    ichoice = [name for name, module in getmembers(io, ismodule)]
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
                export_list= [-2]
                if args['export'][0] in export_check:
                    export_type = args['export'][0][4:]
                else:
                    print("  Unable to parse export type, using all_atoms\n")
                    export_type = 'atoms'
            else:
                export_list = np.array(args['export'][1:], dtype=np.int64)
                if args['export'][0] in export_check:
                    export_type = args['export'][0]
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
    if args.get('fortran_format')  is not None:
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
    filename = "Path to file containing Bader output"
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
    parser.add_argument('filename', nargs=1, help=filename)
    parser.add_argument('-a', '--atoms', action='store_true', help=a)
    parser.add_argument('-v', '--volume', action='store_true', help=v)
    parser.add_argument('-e', '--export', nargs='+', help=e)
    parser.add_argument('-d', '--density-write', action='store_true', help=d)
    parser.add_argument('-f', '--fortran-format', action='count', help=f)
    parser.add_argument('-r', '--recast', action='store_true', help=r)
    args = vars(parser.parse_args())

    with open(args['filename'][0], '+rb') as f:
        bader = load(f)

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
                    export_type = args['export'][0]
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
        if args['fortran_format'] is not None:
            bader.fortran_format = args['fortran_format'] % 3
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
    with open(__config__,'w') as f:
        config.write(f)
    print("Done.")


def JIT_caching():
    with tqdm_wrap(total=18, desc="Caching JIT'd functions:", file=sys.stderr) as pbar:
        # cache factor_3d
        _ = factor_3d(18)
        pbar.update(1)
        # set up arrays
        idx = np.zeros(3, dtype=np.int64, order='C')
        i_c = np.zeros(1, dtype=np.int64)
        positive_len = np.ascontiguousarray([10, 10, 10], dtype=np.int64)
        extend_len = np.ascontiguousarray([20, 20, 20], dtype=np.int64)
        atoms = np.ascontiguousarray([[.25, .25, .25], [.75, .75, .75]])
        density = np.zeros((5, 5, 5), dtype=np.float64, order='C')
        for i in np.ndindex(5, 5, 5):
            density[i] = np.prod(i)
        density = np.ascontiguousarray(np.pad(density, (0, 4), 'reflect'))
        density = np.ascontiguousarray(np.pad(density, (0, 9), 'reflect'))
        lattice = np.ascontiguousarray([[2, 0, 0], [2, 1, 0], [-5.2, 0.7, 8.1]])
        voxel_lattice = lattice / density.shape
        d = np.zeros((3, 3, 3, 3), dtype=np.float64)
        d[1, :, :] += voxel_lattice[0]
        d[2, :, :] -= voxel_lattice[0]
        d[:, 1, :] += voxel_lattice[1]
        d[:, 2, :] -= voxel_lattice[1]
        d[:, :, 1] += voxel_lattice[2]
        d[:, :, 2] -= voxel_lattice[2]
        d = d**2
        d = np.sum(d, axis=3)
        d[d != 0] = d[d != 0]**-.5
        inv_l = np.linalg.inv(voxel_lattice)
        T_grad = np.matmul(inv_l.T, inv_l)
        # cache array_assign
        atoms = array_assign(atoms, 1, 1)
        pbar.update(1)
        # cache array_merge
        _ = array_merge(atoms, atoms)
        pbar.update(1)
        # cache vacuum_assign
        volumes = vacuum_assign(density, 1)
        pbar.update(1)
        # cache volume_extend
        large = volume_extend(volumes, positive_len, extend_len)
        pbar.update(1)
        # cache volume_merge
        volume_merge(large, volumes, idx, (5, 5, 5))
        pbar.update(1)
        # cache volume_offset
        volume_offset(large, 3, 1)
        pbar.update(1)
        # set up Bader class
        density_dict = {
            'charge': density
        }
        file_info = {
            'out_dest': os.devnull,
            'voxel_offset': (0, 0, 0),
        }
        bader = Bader(density_dict, lattice, atoms, file_info)
        bader.threads = 8
        with nostdout():
            bader()
        pbar.update(10)
        bader.method = 'ongrid'
        with nostdout():
            bader()
        pbar.update(1)
