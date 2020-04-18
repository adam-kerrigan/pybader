import numpy as np
from ase.build import molecule
from gpaw import GPAW
from pybader.io import gpaw
from pybader.interface import Bader
import matplotlib.pyplot as plt

# THIS SCRIPT REQUIRES GPAW AND MATPLOTLIB

# Run GPAW calculation
atoms = molecule('H2O')
atoms.center(vacuum=3.5)
atoms.calc = GPAW(h=0.17, txt='h2o.txt')
atoms.get_potential_energy()

# Run Bader
bader = Bader(*gpaw.read_obj(atoms.calc))
bader(threads=4)

# Take slice through atoms
x = bader.density.shape[0] // 2
density = bader.density[x]
vol_nums = bader.atoms_volumes[x]

# set up grid from with atoms at centre
x0, y0, z0 = atoms.positions[0]
y = np.linspace(0, atoms.cell[1, 1], density.shape[0], endpoint=False) - y0
z = np.linspace(0, atoms.cell[2, 2], density.shape[1], endpoint=False) - z0

# plot
plt.figure(figsize=(5, 5))
plt.contourf(z, y, density, np.linspace(0.07, 6, 15))
plt.contour(z, y, vol_nums, [0.5], colors='k')
plt.axis(xmin=-2, xmax=2, ymin=-2, ymax=2)
plt.savefig('h2o-bader.png')
