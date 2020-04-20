from pybader.interface import Bader
import sys
from time import time

# An implementation of the Bader.__call__ function for the speed profile
# as an example of the availbale methods.

# Useage: python bader.py path/to/density

t0 = time()
bader = Bader.from_file(sys.argv[1], spin_flag=True)
bader.load_config('speed')
bader.threads = 8
bader.spin_flag = True  # loading speed config resets all config vars
bader.volumes_init()
bader.bader_calc()
bader.bader_to_atom_distance()
bader.refine_volumes(bader.atoms_volumes)
bader.threads = 1
bader.min_surface_distance()
bader.sum_volumes()
print('\n  Writing output file: ', end='')
if bader.output == 'pickle':
    bader.to_file()
print('Done.')
print(f"Time taken: {time() - t0:.3f}s")
