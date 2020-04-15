import numpy as np
import matplotlib.pyplot as plt
from pybader.interface import Bader
from pybader.utils import nostdout
import sys
import os

# THIS SCRIPT REQUIRES MATPLOTLIB
# Useage: python compare_methods.py density_file.file_type

# Load file and run full refinment
bader = Bader.from_file(sys.argv[1], threads=20)
bader.info['out_dest'] = os.devnull
bader.refine_mode = ('all', -1)
bader()
correct_charge = bader.atoms_charge.copy()

# Initialise lists for each method and total and max
ng_norm_t = []
ng_norm_m = []
og_norm_t = []
og_norm_m = []
ng_speed_t = []
ng_speed_m = []
og_speed_t = []
og_speed_m = []

# Iterate over the number refinement iterations
for i in range(2):
    bader.refine_mode = ('changed', i)
    bader.speed_flag = False
    bader.method = 'neargrid'
    with nostdout():
        bader()
    ng_norm_t.append(np.sum(np.abs(correct_charge - bader.atoms_charge)))
    ng_norm_m.append(np.max(np.abs(correct_charge - bader.atoms_charge)))
    bader.method = 'ongrid'
    with nostdout():
        bader()
    og_norm_t.append(np.sum(np.abs(correct_charge - bader.atoms_charge)))
    og_norm_m.append(np.max(np.abs(correct_charge - bader.atoms_charge)))
    bader.method = 'neargrid'
    bader.speed_flag = True
    with nostdout():
        bader()
    ng_speed_t.append(np.sum(np.abs(correct_charge - bader.atoms_charge)))
    ng_speed_m.append(np.max(np.abs(correct_charge - bader.atoms_charge)))
    bader.method = 'ongrid'
    with nostdout():
        bader()
    og_speed_t.append(np.sum(np.abs(correct_charge - bader.atoms_charge)))
    og_speed_m.append(np.max(np.abs(correct_charge - bader.atoms_charge)))

# Plot the output
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(ng_norm_t, label='neargid total difference')
ax.plot(ng_norm_m, label='neargid max difference')
ax.plot(og_norm_t, label='ongid total difference')
ax.plot(og_norm_m, label='ongid max difference')
ax.plot(ng_speed_t, label='neargid --speed total difference')
ax.plot(ng_speed_m, label='neargid --speed max difference')
ax.plot(og_speed_t, label='ongid --speed total difference')
ax.plot(og_speed_m, label='ongid --speed max difference')
ax.set_ylabel('log(Charge Difference)')
ax.set_xlabel('Refinement Iterations')
ax.set_yscale('log')
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])
# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=2, fancybox=True, shadow=True)
plt.savefig('bader_log.png')
