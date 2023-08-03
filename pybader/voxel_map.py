from bader_python import get_voxelmap
from multiprocessing import cpu_count


class VoxelMap:
    def __init__(
        self,
        reference,
        lattice,
        positions,
        voxel_origin=[0, 0, 0],
        visible_bar=True,
        weight_tolerance=1e-8,
        maximum_distance=0.1,
        threads=None,
        vacuum_tolerance=1e-6,
    ) -> None:
        if threads is None:
            threads = min(cpu_count(), 12)
        self.lattice = lattice
        self.voxel_origin = voxel_origin
        self.positions = positions
        grid = reference.shape
        self.saddles, self.voxelmap, self.weightmap = get_voxelmap(
            reference.ravel(),
            grid,
            lattice,
            positions,
            voxel_origin,
            visible_bar,
            weight_tolerance,
            maximum_distance,
            threads,
            vacuum_tolerance,
        )
