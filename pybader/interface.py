"""The user interface for accessing the Bader calulation.

Contains the Bader class, dictionaries containing the attributes of the Bader
class along with their types and a config file converter.
"""
import numpy as np
from pickle import dump
from ast import literal_eval
from configparser import ConfigParser
from pybader import (
        io,
        __config__,
)
from pybader.utils import (
        vacuum_assign,
        atom_assign,
        charge_sum,
        volume_mask,
)
from pybader.thread_handlers import (
        bader_calc,
        assign_to_atoms,
        refine,
        surface_distance,
        dtype_calc,
)
from inspect import (
        getmembers,
        ismodule,
)


# Dictionary containing the private attributes and types of the Bader class
private_attributes = {
        '_density': np.ndarray,
        '_lattice': np.ndarray,
        '_atoms': np.ndarray,
        '_file_info': dict,
        '_bader_maxima': np.ndarray,
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
        self.density = self.charge if self.charge is not None else self.spin
        self.reference = self.density
        for key, value in python_config().items():
            self.__setattr__(key, value)
        for key, value in kwargs.items():
            self.__setattr__(key, value)

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
            io_packages = getmembers(io, ismodule)
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
        file_conf = {k: v for k, v in kwargs.items() if k in io.chgcar.__args__}
        return cls(*io.chgcar.read(filename, **file_conf), **kwargs)

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

    def __call__(self, **kwargs):
        """Run the Bader calculation on self.

        args:
            keyword arguements accepted are listed in the __slots__ attribute
        """
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        self.volumes_init()
        self.bader_calc()
        if not self.speed_flag:
            self.refine_volumes(self.bader_volumes)
            self.bader_charge = np.zeros(self.bader_maxima.shape[0] + 1,
                    dtype=np.float64)
            self.bader_volume = np.zeros(self.bader_maxima.shape[0] + 1,
                    dtype=np.float64)
            charge_sum(self.bader_charge, self.bader_volume, self.voxel_volume,
                    self.density, self.bader_volumes)
            if self.spin_flag:
                self.bader_spin = np.zeros(self.bader_maxima.shape[0] + 1,
                        dtype=np.float64)
                self.bader_volume = np.zeros(self.bader_maxima.shape[0] + 1,
                        dtype=np.float64)
                charge_sum(self.bader_spin, self.bader_volume,
                        self.voxel_volume, self.density, self.bader_volumes)
        self.bader_to_atom_distance()
        if self.speed_flag:
            self.refine_volumes(self.atoms_volumes)
            del(self.bader_volumes)
        self.min_surface_distance()
        self.atoms_charge = np.zeros(self.atoms.shape[0] + 1, dtype=np.float64)
        self.atoms_volume = np.zeros(self.atoms.shape[0] + 1, dtype=np.float64)
        charge_sum(self.atoms_charge, self.atoms_volume, self.voxel_volume,
                self.density, self.atoms_volumes)
        if self.spin_flag:
            self.atoms_spin = np.zeros(self.atoms.shape[0] + 1,
                    dtype=np.float64)
            self.atoms_volume = np.zeros(self.atoms.shape[0] + 1,
                    dtype=np.float64)
            charge_sum(self.atoms_spin, self.atoms_volume, self.voxel_volume,
                    self.spin, self.atoms_volumes)
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
        dtype = dtype_calc(np.prod(self.density.shape))
        volumes = np.zeros(self.density.shape, dtype=dtype)
        try:
            vacuum_tol = self.vacuum_tol * self.lattice_volume
            volumes = vacuum_assign(self.reference, vacuum_tol)
        except TypeError:
            pass
        self.bader_volumes = volumes

    def bader_calc(self):
        """Launch the thread handler for the Bader calculation.
        """
        self.bader_maxima, self.bader_volumes = bader_calc(self.method, 
                self.reference, self.bader_volumes, self.distance_matrix, 
                self.T_grad, self.threads)

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

    def min_surface_distance(self):
        """Launch the thread handler for calculating the min. surface distance.
        """
        atoms = self.atoms - self.voxel_offset
        self.atoms_surface_distance = surface_distance(self.reference,
                self.atoms_volumes, self.lattice, atoms, self.threads)

    def results(self, volume_flag=False):
        """Format the results from a calcution as a string.

        args:
            volume_flag: whether to do this for atoms or Bader volumes
        returns:
            formatted string
        """
        if volume_flag:
            charge = self.bader_charge
            volume = self.bader_volume
            if self.spin_flag:
                spin = self.bader_spin
            atoms = self.bader_maxima
            dist = self.bader_distance
        else:
            charge = self.atoms_charge
            volume = self.atoms_volume
            if self.spin_flag:
                spin = self.atoms_spin
            atoms = self.atoms_fractional
            dist = self.atoms_surface_distance
        m = charge[:-1] != 0
        charge_width = np.max(np.log10(np.abs(charge[:-1][m])) + 1)
        charge_width = max([int(charge_width), 1]) + 9
        m = volume[:-1] != 0
        volume_width = np.max(np.log10(volume[:-1][m]) + 1)
        volume_width = max([int(volume_width), 1]) + 8
        if self.spin_flag:
            m = spin[:-1] != 0
            spin_width = np.max(np.log10(np.abs(spin[:-1][m])) + 1)
            spin_width = max([int(spin_width), 1]) + 9
        atom_width = (np.log10(atoms.shape[0]) + 1).astype(int)
        atom_width = max([atom_width, 1]) + 1
        m = dist != 0
        dist_width = np.max(np.log10(dist[m]) + 1)
        dist_width = max([int(dist_width), 1]) + 8
        h = ['#', ' a', ' b', ' c', 'Charge', 'Spin', 'Volume', 'Dist']
        if not atom_width % 2:
            h[0] = ' #'
        if not charge_width % 2:
            h[4] = ' Charge'
        if self.spin_flag and not spin_width % 2:
            h[5] = ' Spin'
        if volume_width % 2:
            h[6] = ' Volume'
        if dist_width % 2:
            h[7] = ' Dist'
        header_format = f"{{:^{atom_width}s}}" + "{:^8s}" * 3
        if self.spin_flag:
            header_format += f"{{:^{charge_width}s}}{{:^{spin_width}s}}"
        else:
            header_format += f"{{:^{charge_width}s}}"
            h.pop(5)
        header_format += f"{{:^{volume_width}}}"
        header_format += f"{{:^{dist_width}s}}\n"

        header = header_format.format(*h)
        line = '-' * len(header) + '\n'
        table = ''
        for i, atom in enumerate(atoms):
            if volume_flag and np.abs(charge[i]) < self.bader_volume_tol:
                continue
            a, b, c = atom
            table += f"{i:>{atom_width}}{a:> 8.4f}{b:> 8.4f}{c:> 8.4f}"
            table += f"{charge[i]:> {charge_width}.6f}"
            if self.spin_flag:
                table += f"{spin[i]:> {spin_width}.6f}"
            table += f"{volume[i]:>{volume_width}.6f}"
            table += f"{dist[i]:>{dist_width}.6f}"
            table += '\n'
        footer = ''
        tot_charge = np.sum(charge)
        elec_width = np.log10(np.abs(tot_charge)) + 8
        footer_width = int(elec_width)
        if self.vacuum_tol is not None:
            vac_items = [charge[-1], volume[-1]]
            if self.spin_flag:
                vac_items.append(spin[-1] * 100)
            vac_width = np.max(np.log10(np.abs(vac_items))).astype(int) + 8
            if vac_width > footer_width:
                footer_width = vac_width
            footer = f" Vacuum Charge:"
            footer += f"{charge[-1]:>{footer_width + 6}.4f}\n"
            if self.spin_flag:
                footer += f" Vacuum Spin:"
                footer += f"{spin[-1]:>{footer_width + 8}.4f}\n"
            footer += f" Vacuum Volume:"
            footer += f"{volume[-1]:>{footer_width + 6}.4f}\n"
        footer += f" Number of Electrons:"
        footer += f"{np.sum(tot_charge):>{footer_width}.4f}"
        return header + line + table + line + footer

    def load_config(self, key='DEFAULT'):
        """Load a profile from the config.ini file.

        args:
            key: name of the profile to load
        """
        for k, value in python_config(key=key).items():
            self.__setattr__(k, value)

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
        self.info['write_function'](f"Bader-{self.export_mode[0]}-{num}",
                self.atoms, self.lattice, density, self.info,
                prefix=self.info['prefix'])

    def write_density(self):
        """Write the full density as stored in the density dict.

        Use info['fortran_format'] to set the 'fortran-ness' of the output.
        """
        self._file_info['comment'] = f"Full charge density output\n"
        self._file_info['fortran_format'] = self.fortran_format
        self.info['write_function'](f"{self.info['filename']}", self.atoms,
                self.lattice, self._density, self.info, suffix='')
