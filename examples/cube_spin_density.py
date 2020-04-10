from pybader.interface import Bader
from pybader.io import cube
import sys

# Usage: python cube_spin_density.py charge_density.cube spin_density.cube


# Read in both files using the charge density to initialise the Bader class
bader = Bader.from_file(sys.argv[1])
density, _, _, _ = cube.read(sys.argv[2])

# Set the spin attribute to the density of the spin density file and set spin on
bader.spin = density['charge']
bader.spin_flag = True

# run the bader calculation
bader()
