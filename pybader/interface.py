"""The user interface for accessing the Bader calulation.

Contains the Bader class, dictionaries containing the attributes of the Bader
class along with their types and a config file converter.
"""
from ast import literal_eval
from configparser import ConfigParser
from inspect import getmembers, ismodule
from pickle import dump
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd

from pybader import __config__, io
from pybader.thread_handlers import (assign_to_atoms, bader_calc, dtype_calc,
                                     refine, surface_distance)
from pybader.utils import atom_assign, charge_sum, vacuum_assign, volume_mask

# Dictionary containing the private attributes and types of the Bader class
private_attributes = {
    '_density': np.ndarray,
    '_lattice': np.ndarray,
    '_atoms': np.ndarray,
    '_file_info': dict,
    '_bader_maxima': np.ndarray,
    '_vacuum_charge': float,
    '_vacuum_volume': float,
    '_dataframe': pd.DataFrame
}


# Dictionary containing the configurable attributes and types of the Bader class
config_attributes = {
    'method': str,
    'refine_method': str,
    'vacuum_tol': (type(None), float),
    'refine_mode': (str, int),
    'bader_volume_tol': (type(None), float),
    'export_mode': (type(None), str, int),
    'prefix': str,
    'output': str,
    'threads': int,
    'fortran_format': int,
    'speed_flag': bool,
    'spin_flag': bool,
}


# Dictionary containing the accessible attributes and types of the Bader class
properties = {
    'density': property,
    'reference': property,
    'bader_charge': np.ndarray,
    'bader_volume': np.ndarray,
    'bader_spin': np.ndarray,
    'bader_volumes': np.ndarray,
    'bader_atoms': np.ndarray,
    'bader_distance': np.ndarray,
    'atoms_charge': np.ndarray,
    'atoms_volume': np.ndarray,
    'atoms_spin': np.ndarray,
    'atoms_volumes': np.ndarray,
    'atoms_surface_distance': np.ndarray,
}


def python_config(config_file=__config__, key='DEFAULT'):
    """Converts a profile in the config.ini file to the correct python type.

    args:
        config_file: the location of the config file
        key: the name of the profile to load
    returns:
        dictionary representation of the config with evaluated values
    """
    config = ConfigParser()
    with open(config_file, 'r') as f:
        config.read_file(f)
    if not key in config:
        print("  No config for {key} found")
    python_config = {}
    for k in config[key]:
        if k in config_attributes:
            try:
                python_config[k] = literal_eval(config[key].get(k))
            except ValueError as e:
                if config_attributes[k] is str:
                    python_config[k] = config[key].get(k)
                else:
                    raise e
        else:
            raise AttributeError(f"  Unknown keyword in config.ini: {k}")
        if not isinstance(python_config[k], config_attributes[k]):
            e = f"  {k} has wrong type: {type(k)} != {config_attributes[k]}"
            if hasattr(python_config[k], '__iter__'):
                for t in python_config[k]:
                    if not isinstance(t, config_attributes[k]):
                        raise TypeError(e)
            else:
                raise TypeError(e)
    return python_config


class Bader:
    f"""Class for easily running Bader calculations.

    Loads default config from '{__config__}'

    args:
        density_dict: dictionary with any or all of the keys 'charge', 'spin'
                      the value associated with these keys should be an ndarray
                      of the respective density in (rho * lattice volume) units
        lattice: the lattice of the periodic cell with lattice vectors on the
                 first axis and in cartesian coordinates
        atoms: the coordinates, in cartesian, of the atoms in the cell
        file_info: dictionary of information about the file read in, including
                   filename, file_type, prefix (directory path), voxel_offset
                   and write function
        other keyword arguements accepted are listed in the __slots__ attribute
    """
    __slots__ = [
        *private_attributes.keys(),
        *config_attributes.keys(),
        *properties.keys(),
    ]

    def __init__(self, density_dict, lattice, atoms, file_info, **kwargs):
        """Initialise the class with default config and then apply any kwargs.
        """
        self._density = density_dict
        self._lattice = lattice
        self._atoms = atoms
        self._file_info = file_info
        self._dataframe = None
        self.density = self.charge if self.charge is not None else self.spin
        self.reference = self.density
        self.load_config()
        self.apply_config(kwargs)

    @classmethod
    def from_file(cls, filename, file_type=None, **kwargs):
        """Class method for initialising from file

        args:
            filename: the location of the file
            file_type: the type of the file (useful if filename doesn't contain
                       obvious type)
            other keyword arguments are file-type specific and are listed in the
            __args__ variable in the respective module
        """
        if file_type is not None:
            file_type = file_type.lower()
            for f_type, f_method in getmembers(io, ismodule):
                if f_type == file_type:
                    io_ = f_method
            file_conf = {k: v for k, v in kwargs.items() if k in io_.__args__}
            return cls(*io_.read(filename, **file_conf), **kwargs)
        else:
            io_packages = (p for p in getmembers(io, ismodule)
                           if p[1].__extensions__ is not None)
            for package in io_packages:
                for ext in package[1].__extensions__:
                    io_ = package[1] if ext in filename.lower() else None
                    if io_ is not None:
                        file_conf = {
                            k: v for k, v in kwargs.items()
                            if k in io_.__args__
                        }
                        return cls(*io_.read(filename, **file_conf), **kwargs)
        print("  No clear file type found; file will be read as chgcar.")
        file_conf = {k: v for k, v in kwargs.items() if k in io.vasp.__args__}
        return cls(*io.vasp.read(filename, **file_conf), **kwargs)

    @classmethod
    def from_dict(cls, d):
        """Create class entirely from dictonary.
        """
        atoms = d.pop('_atoms')
        lattice = d.pop('_lattice')
        density = d.pop('_density')
        file_info = d.pop('_file_info')
        cls(density, lattice, atoms, file_info, **d)

    @property
    def as_dict(self):
        """Convert class to dictionary.
        """
        d = {}
        for key in self.__slots__:
            try:
                d[key] = getattr(self, key)
            except AttributeError:
                pass
        return d

    @property
    def info(self):
        """Access the file_info dictionary.
        """
        return self._file_info

    @property
    def charge(self):
        """Access the charge density in the density dictionary.
        """
        return self._density.get('charge', None)

    @property
    def spin(self):
        """Access the spin density in the density dictionary.
        """
        return self._density.get('spin', None)

    @property
    def spin_bool(self):
        """Whether to perfom on the spin density also.

        This should only return true if spin is not None.
        """
        return self.spin_flag if self.spin is not None else False

    @spin_bool.setter
    def spin_bool(self, flag):
        """Set the spin flag.
        """
        self.spin_flag = flag

    @property
    def lattice(self):
        """Access the lattice describing the periodic cell.
        """
        return self._lattice

    @property
    def lattice_volume(self):
        """Calculate the volume of the lattice.
        """
        v = np.dot(self.lattice[0], np.cross(*self.lattice[1:]))
        return np.abs(v)

    @property
    def distance_matrix(self):
        """Calculate a matrix of distances relating to steps of size index.

        matrix is size 3x3x3 however index (2, 2, 2) is not a step of 2 in each
        direction but instead a step of (-1, -1, -1).
        """
        d = np.zeros((3, 3, 3, 3), dtype=np.float64)
        d[1, :, :] += self.voxel_lattice[0]
        d[2, :, :] -= self.voxel_lattice[0]
        d[:, 1, :] += self.voxel_lattice[1]
        d[:, 2, :] -= self.voxel_lattice[1]
        d[:, :, 1] += self.voxel_lattice[2]
        d[:, :, 2] -= self.voxel_lattice[2]
        d = d**2
        d = np.sum(d, axis=3)
        d[d != 0] = d[d != 0]**-.5
        return d

    @property
    def voxel_lattice(self):
        """A lattice desctibing the dimensions of a voxel.
        """
        return np.divide(self.lattice, self.density.shape)

    @property
    def voxel_volume(self):
        """Calculate the volume of a single voxel.
        """
        return self.lattice_volume / np.prod(self.density.shape)

    @property
    def voxel_offset(self):
        """The position of the charge described by the voxel to it's origin.
        """
        return np.dot(self.voxel_offset_fractional, self.voxel_lattice)

    @property
    def voxel_offset_fractional(self):
        """The voxel offset in fractional coordinates w.r.t. it's own lattice.
        """
        return self.info['voxel_offset']

    @property
    def T_grad(self):
        """The transform matrix for converting voxel step to gradient step.
        """
        inv_l = np.linalg.inv(self.voxel_lattice)
        return np.matmul(inv_l.T, inv_l)

    @property
    def atoms(self):
        """Access the atoms.
        """
        return self._atoms

    @atoms.setter
    def atoms(self, array):
        """Set the atoms enforcing shape (len(atoms), 3).
        """
        array = np.asarray(array).flatten()
        array = array.reshape(array.shape[0] // 3, 3)
        self._atoms = np.ascontiguousarray(array)

    @property
    def atoms_fractional(self):
        """Return the atoms in fractional coordinates.
        """
        return np.dot(self.atoms, np.linalg.inv(self.lattice))

    @property
    def bader_maxima(self):
        """The location of the Bader maxima in cartesian coordinates.
        """
        return np.dot(self.bader_maxima_fractional, self.lattice)

    @bader_maxima.setter
    def bader_maxima(self, maxima):
        """Set the location of the Bader maxima.
        """
        maxima = np.add(maxima, self.voxel_offset_fractional)
        maxima = np.divide(maxima, self.density.shape)
        self._bader_maxima = np.ascontiguousarray(maxima)

    @property
    def bader_maxima_fractional(self):
        """Return the Bader maxima in fractional coordinates.
        """
        try:
            return self._bader_maxima
        except AttributeError:
            print("  ERROR: bader_maxima not yet set.")
            return

    @property
    def vacuum_charge(self):
        return getattr(self, '_vacuum_charge', 0.)

    @vacuum_charge.setter
    def vacuum_charge(self, value):
        self._vacuum_charge = value

    @property
    def vacuum_volume(self):
        return getattr(self, '_vacuum_volume', 0.)

    @vacuum_volume.setter
    def vacuum_volume(self, value):
        self._vacuum_volume = value

    @property
    def dataframe(self):
        if self._dataframe is None:
            a = pd.Series(self.atoms_fractional[:, 0], name='a')
            b = pd.Series(self.atoms_fractional[:, 1], name='b')
            c = pd.Series(self.atoms_fractional[:, 2], name='c')
            charge = pd.Series(self.atoms_charge, name='Charge')
            volume = pd.Series(self.atoms_volume, name='Volume')
            distance = pd.Series(self.atoms_surface_distance, name='Distance')
            if self.spin_bool:
                spin = pd.Series(self.atoms_spin, name='Spin')
            if not self.speed_flag:
                a = a.append(
                    pd.Series(self.bader_maxima_fractional[:, 0], name='a'),
                )
                b = b.append(
                    pd.Series(self.bader_maxima_fractional[:, 1], name='b'),
                )
                c = c.append(
                    pd.Series(self.bader_maxima_fractional[:, 2], name='c'),
                )
                charge = charge.append(
                    pd.Series(self.bader_charge, name='Charge'),
                )
                volume = volume.append(
                    pd.Series(self.bader_volume, name='Volume'),
                )
                distance = distance.append(
                    pd.Series(self.bader_distance, name='Distance'),
                )
                if self.spin_bool:
                    spin = spin.append(
                        pd.Series(self.bader_spin, name='Spin'),
                    )
            if self.spin_bool:
                self.dataframe = pd.concat(
                    [a, b, c, charge, spin, volume, distance],
                    axis=1)
            else:
                self.dataframe = pd.concat([a, b, c, charge, volume, distance],
                                           axis=1)
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df):
        self._dataframe = df

    def __call__(self, **kwargs):
        """Run the Bader calculation on self.

        args:
            keyword arguements accepted are listed in the __slots__ attribute
        """
        self.apply_config(kwargs)
        self.volumes_init()
        self.bader_calc()
        if not self.speed_flag:
            self.refine_volumes(self.bader_volumes)
            self.sum_volumes(bader=True)
        self.bader_to_atom_distance()
        if self.speed_flag:
            self.refine_volumes(self.atoms_volumes)
            del(self.bader_volumes)
        self.min_surface_distance()
        self.sum_volumes()
        if self.export_mode is not None:
            print(f"\n  Writing Bader {self.export_mode[0]} to file:")
            if self.export_mode[0] == 'volumes':
                if self.export_mode[1] == -2:
                    for vol_num in range(self.bader_maxima.shape[0]):
                        self.write_volume(vol_num)
                    if self.vacuum_tol is not None:
                        self.write_volume(-1)
                else:
                    for vol_num in self.export_mode[1]:
                        self.write_volume(vol_num)
            elif self.export_mode[0] == 'atoms':
                if self.export_mode[1] == -2:
                    for vol_num in range(self.atoms.shape[0]):
                        self.write_volume(vol_num)
                    if self.vacuum_tol is not None:
                        self.write_volume(-1)
                else:
                    for vol_num in self.export_mode[1]:
                        self.write_volume(vol_num)
        print('\n  Writing output file: ', end='')
        if self.output == 'pickle':
            self.to_file()
        elif self.output == 'dat':
            fn = self.prefix + self.info['filename']
            with open(fn + '-atoms.dat', 'w') as f:
                f.write(self.results())
            if not self.speed_flag:
                with open(fn + '-volumes.dat', 'w') as f:
                    f.write(self.results(volume_flag=True))
        print('Done.')

    def volumes_init(self):
        """Initialise the bader_volumes array using vacuum_tol.
        """
        dtype = dtype_calc(-np.prod(self.density.shape))
        volumes = np.zeros(self.density.shape, dtype=dtype)
        try:
            vacuum_tol = np.float64(self.vacuum_tol)
            volumes, vacuum_charge, vacuum_volume = vacuum_assign(
                self.reference, volumes, vacuum_tol, self.density,
                self.voxel_volume)
            self.vacuum_charge = vacuum_charge
            self.vacuum_volume = vacuum_volume
        except (ValueError, TypeError) as e:
            print(f"  VACUUM_TOL ERROR: {self.vacuum_tol} is not float")
            print(f"  {e}")
        finally:
            self.bader_volumes = volumes

    def bader_calc(self):
        """Launch the thread handler for the Bader calculation.
        """
        self.bader_maxima, self.bader_volumes = bader_calc(
            self.method, self.reference, self.bader_volumes,
            self.distance_matrix, self.T_grad, self.threads
        )

    def bader_to_atom_distance(self):
        """Launch the thread handler for assigning Bader volumes to atoms.
        """
        ret = assign_to_atoms(self.bader_maxima, self.atoms, self.lattice,
                              self.bader_volumes, self.threads)
        self.bader_atoms, self.bader_distance, self.atoms_volumes = ret

    def refine_volumes(self, volumes):
        """Launch the thread handler for refining Bader and/or atom volumes.
        """
        refine(self.refine_method, self.refine_mode, self.reference, volumes,
               self.distance_matrix, self.T_grad, self.threads)

    def sum_volumes(self, bader=False):
        """Sum the density and volume in the Bader volumes/atoms.

        args:
            bader: bool for doing bader volumes (True) or atoms (False)
        """
        if bader:
            self.bader_charge = np.zeros(self.bader_maxima.shape[0],
                                         dtype=np.float64)
            self.bader_volume = np.zeros(self.bader_maxima.shape[0],
                                         dtype=np.float64)
            charge_sum(self.bader_charge, self.bader_volume, self.voxel_volume,
                       self.density, self.bader_volumes)
            if self.spin_bool:
                self.bader_spin = np.zeros(self.bader_maxima.shape[0],
                                           dtype=np.float64)
                self.bader_volume = np.zeros(self.bader_maxima.shape[0],
                                             dtype=np.float64)
                charge_sum(self.bader_spin, self.bader_volume,
                           self.voxel_volume, self.spin, self.bader_volumes)
        else:
            self.atoms_charge = np.zeros(self.atoms.shape[0],
                                         dtype=np.float64)
            self.atoms_volume = np.zeros(self.atoms.shape[0],
                                         dtype=np.float64)
            charge_sum(self.atoms_charge, self.atoms_volume, self.voxel_volume,
                       self.density, self.atoms_volumes)
            if self.spin_bool:
                self.atoms_spin = np.zeros(self.atoms.shape[0],
                                           dtype=np.float64)
                self.atoms_volume = np.zeros(self.atoms.shape[0],
                                             dtype=np.float64)
                charge_sum(self.atoms_spin, self.atoms_volume,
                           self.voxel_volume, self.spin, self.atoms_volumes)

    def min_surface_distance(self):
        """Launch the thread handler for calculating the min. surface distance.
        """
        atoms = self.atoms - self.voxel_offset
        self.atoms_surface_distance = surface_distance(
            self.reference, self.atoms_volumes, self.lattice,
            atoms, self.threads
        )

    def results(self, volume_flag=False):
        """Format the results from a calcution as a string.

        args:
            volume_flag: whether to do this for atoms or Bader volumes
        returns:
            formatted string
        """
        if volume_flag:
            df = self.dataframe[self.atoms.shape[0]:]
            df = df[df['Charge'] > self.bader_volume_tol]
        else:
            df = self.dataframe[:self.atoms.shape[0]]
        df_text = df.to_string(float_format='{:.6f}'.format, justify='center')
        df_text = df_text.split('\n')
        for i, line in enumerate(df_text):
            df_text[i] = ' ' + line + '\n'
        df_text.insert(1, '-' * len(df_text[0]) + '\n')
        df_text.append('-' * len(df_text[0]) + '\n')
        df_text = ''.join(df_text)
        footer = ''
        tot_charge = df['Charge'].sum()
        elec_width = np.log10(np.abs(tot_charge)) + 8
        footer_width = int(elec_width)
        if self.vacuum_tol is not None:
            vac_items = [self.vacuum_charge, self.vacuum_volume]
            vac_width = np.max(np.log10(np.abs(vac_items))).astype(int) + 8
            if vac_width > footer_width:
                footer_width = vac_width
            footer = f" Vacuum Charge:"
            footer += f"{self.vacuum_charge:>{footer_width + 6}.4f}\n"
            footer += f" Vacuum Volume:"
            footer += f"{self.vacuum_volume:>{footer_width + 6}.4f}\n"
        footer += f" Number of Electrons:"
        footer += f"{tot_charge:>{footer_width}.4f}"
        return df_text + footer

    def apply_config(self, d):
        """Apply a config dictionary to the class.

        args:
            d: dictionary of settings to be applied

        -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        Future: This should probably type check
        """
        for k, value in d.items():
            self.__setattr__(k, value)

    def load_config(self, key='DEFAULT'):
        """Load a profile from the config.ini file.

        args:
            key: name of the profile to load
        """
        self.apply_config(python_config(key=key))

    def to_file(self):
        """Pickle self and store at prefix + bader.p or 'out_dest' in info.
        """
        filename = self.info.get('out_dest', self.prefix + 'bader.p')
        with open(filename, '+wb') as f:
            dump(self, f)

    def write_volume(self, vol_num):
        """Write a specific Bader volume or atom to file.
        """
        density = {}
        if self.export_mode[0] == 'volumes':
            volumes = self.bader_volumes
        else:
            volumes = self.atoms_volumes
        if self.charge is not None:
            density['charge'] = volume_mask(volumes, self.charge, vol_num)
        if self.spin is not None:
            density['spin'] = volume_mask(volumes, self.spin, vol_num)
        if vol_num != -1:
            num = vol_num
        else:
            num = 'vacuum'
        self._file_info['comment'] = f"Bader {self.export_mode[0]}: {num}\n"
        self._file_info['fortran_format'] = self.fortran_format
        self.info['write_function'](
            f"Bader-{self.export_mode[0]}-{num}", self.atoms, self.lattice,
            density, self.info, prefix=self.info['prefix']
        )

    def write_density(self):
        """Write the full density as stored in the density dict.

        Use info['fortran_format'] to set the 'fortran-ness' of the output.
        """
        self._file_info['comment'] = f"Full charge density output\n"
        self._file_info['fortran_format'] = self.fortran_format
        self.info['write_function'](f"{self.info['filename']}", self.atoms,
                                    self.lattice, self._density, self.info, suffix='')
