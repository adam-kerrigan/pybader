from pybader.interface import Bader

VACUUM_TOL = 1E-1
bader = Bader.from_file("CHGCAR")
bader(speed_flag=True)

for i in range(10, 0, -1):
    bader.vacuum_tol = i * 1E-4
    bader.volumes_init(bader.atoms_volumes)
    bader.atoms_volumes = bader.bader_volumes
    bader.sum_volumes()
    if bader.vacuum_charge < VACUUM_TOL:
        break
print(bader.results())
print(f" Vacuum Tolerance: {i * 1E-4}")
