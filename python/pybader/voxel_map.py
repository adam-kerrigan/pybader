from pybader import _pybader


class pyVoxelMap(_pybader.PyVoxelMap):
    @classmethod
    def from_pymatgen(cls, cc):
        return cls(
            cc.data["total"],
            cc.structure.lattice.matrix,
            cc.structure.cart_coords,
            [0.0, 0.0, 0.0],
        )

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        keys = ["charge"]
        if ".cube" in filename:
            (density, lattice, positions) = _pybader.read_cube(filename)
            voxel_origin = [0.5, 0.5, 0.5]
        else:
            (density, lattice, positions) = _pybader.read_vasp(filename)
            voxel_origin = [0.0, 0.0, 0.0]
            if len(density) < 3:
                keys.append("spin")
            else:
                keys.append("spin X")
                keys.append("spin Y")
                keys.append("spin Z")
        return (
            cls(
                density[0],
                lattice,
                positions,
                voxel_origin,
                **kwargs,
            ),
            {k: d for (d, k) in zip(density, keys)},
        )
