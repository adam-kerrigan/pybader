use bader::analysis::assign_maxima;
use bader::analysis::{calculate_bader_density, calculate_bader_volumes_and_radii};
use bader::arguments::{App, Args};
use bader::atoms::{Atoms, Lattice};
use bader::methods::{maxima_finder, weight};
use bader::utils::vacuum_index;
use bader::voxel_map::BlockingVoxelMap;
use bader::voxel_map::VoxelMap;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A class for holding relevant information for Bader Charge Analysis.
///
/// The class builds the VoxelMap for a supplied reference density along with
/// the atoms within the structure, the calculated saddle points and parameters
/// of the calculation.
#[pyclass(frozen, subclass)]
pub struct PyVoxelMap {
    voxel_map: VoxelMap,
    atoms: Atoms,
    saddles: Vec<isize>,
    args: Args,
}

#[pymethods]
impl PyVoxelMap {
    /// Create a new PyVoxelMap by running Bader Charge Partitioning.
    #[new]
    #[pyo3(signature = (
        reference,
        lattice,
        positions,
        voxel_origin = [0.0, 0.0, 0.0],
        **kwds
    ))]
    fn new(
        reference: PyReadonlyArray3<f64>,
        lattice: [[f64; 3]; 3],
        positions: Vec<[f64; 3]>,
        voxel_origin: [f64; 3],
        kwds: Option<&PyDict>,
    ) -> Self {
        // check optional flags
        let flags = match kwds {
            None => String::from("bca file.cube"),
            Some(dict) => {
                let mut flags = String::from("bca file.cube");
                if let Some(b) = dict.get_item("hidden_bar") {
                    if b.is_true().unwrap() {
                        flags.push_str(" -x");
                    }
                };
                if let Some(f) = dict.get_item("weight_tolerance") {
                    flags.push_str(format!(" -w {}", f).as_str());
                }
                if let Some(f) = dict.get_item("maximum_distance") {
                    flags.push_str(format!(" -m {}", f).as_str());
                }
                if let Some(u) = dict.get_item("threads") {
                    flags.push_str(format!(" -t {}", u).as_str());
                }
                if let Some(f) = dict.get_item("vacuum_tolerance") {
                    flags.push_str(format!(" -v {}", f).as_str());
                }
                flags
            }
        };
        // type checking on the function will catch the errors
        let args = App::new()
            .parse_args(flags.split_whitespace().collect())
            .unwrap();
        let atoms = Atoms::new(Lattice::new(lattice), positions, String::from(""));
        // This can fail if the inputted density is no contiguous
        // Not a problem if density supplied is formated correctly
        let (grid, reference) = if reference.is_c_contiguous() {
            let g = reference.shape();
            // safe to unwrap from the contiguous check safe to assume 3d from argument type
            ([g[0], g[1], g[2]], reference.as_slice().unwrap())
        } else {
            panic!("Supplied reference density should be c contiguous in memory.")
        };
        let voxel_map = BlockingVoxelMap::new(grid, lattice, voxel_origin);
        let mut index: Vec<usize> = (0..voxel_map.grid.size.total).collect();
        index.sort_unstable_by(|a, b| reference[*b].partial_cmp(&reference[*a]).unwrap());
        // remove from the indices any voxel that is below the vacuum limit
        let vacuum_i = match vacuum_index(reference, &index, args.vacuum_tolerance) {
            Ok(i) => i,
            Err(e) => panic!("{}", e),
        };
        index.truncate(vacuum_i);
        // find the maxima in the system and store them whilst removing them from
        // the index list
        let bader_maxima = maxima_finder(&index, reference, &voxel_map, args.threads, !args.silent);
        // Start a thread-safe progress bar and assign the maxima to atoms
        let atom_map = match assign_maxima(&bader_maxima,
                                       &atoms,
                                       &voxel_map.grid,
                                       &args.maximum_distance,
                                       args.threads,
                                       !args.silent)
    {
        Ok(v) => v,
        Err(e) => panic!(
            "\nBader maximum at {:?}\n is too far away from nearest atom: {} with a distance of {} Ang.",
            e.maximum,
            e.atom + 1,
            e.distance,
        )
    };
        // input the maxima as atoms into the voxel map
        bader_maxima.iter().enumerate().for_each(|(i, maxima)| {
            voxel_map.maxima_store(*maxima, atom_map[i] as isize);
        });
        // calculate the weights leave the saddles for now
        let saddles = weight(
            reference,
            &voxel_map,
            &index,
            args.weight_tolerance,
            !args.silent,
            args.threads,
        );
        // convert into a VoxelMap as the map is filled and no longer needs to block
        let voxel_map = VoxelMap::from_blocking_voxel_map(voxel_map);
        Self {
            voxel_map,
            atoms,
            saddles,
            args,
        }
    }

    /// Calculate the Bader volume and radius for each atom in the VoxelMap.
    ///
    /// Finds the volume and radius of each atom by matching each voxel to the
    /// respective atom. The returned array for volume is of length atoms + 1
    /// as the final element is the volume of voxels regarded as vacuum. There
    /// is no radius for the vacuum.
    fn calculate_volumes_and_radii<'py>(
        &self,
        py: Python<'py>,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let (bv, br) = calculate_bader_volumes_and_radii(
            &self.voxel_map,
            &self.atoms,
            self.args.threads,
            !self.args.silent,
        );
        (bv.into_pyarray(py), br.into_pyarray(py))
    }

    /// Calculate the Bader densities for each atom in the VoxelMap and each density provided.
    ///
    /// Finds the density associated to each atom by matching each voxel to the
    /// respective atom. The returned array for density is of length atoms + 1
    /// as the final element is the density of voxels regarded as vacuum. The
    /// returned dictionary has the same keys as the supplied dictionary.
    fn calculate_bader_densities<'py>(
        &self,
        py: Python<'py>,
        density: PyReadonlyArray3<f64>,
    ) -> &'py PyArray1<f64> {
        // This can fail if the inputted density is no contiguous
        // Not a problem if density supplied is formated correctly
        let density = if density.is_c_contiguous() {
            let g = density.shape();
            let s = &self.voxel_map.grid.size;
            assert_eq!(
                [g[0], g[1], g[2]],
                [s.x as usize, s.y as usize, s.z as usize],
                "Supplied density has different shape to the voxel map.",
            );
            // safe to unwrap from the contiguous check safe to assume 3d from argument type
            density.as_slice().unwrap()
        } else {
            panic!("Supplied reference density should be c contiguous in memory.")
        };
        calculate_bader_density(
            density,
            &self.voxel_map,
            &self.atoms,
            self.args.threads,
            !self.args.silent,
        )
        .into_pyarray(py)
    }

    /// The index of each saddle point in the 3D data.
    ///
    /// An n x 3 sized array of indices, locating saddle points in the reference density, where n
    /// is the number of saddle points.
    #[getter]
    fn saddle_points<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
        PyArray2::from_vec2(
            py,
            &self
                .saddles
                .iter()
                .map(|p| {
                    let grid = &self.voxel_map.grid;
                    let x = (p / (grid.size.y * grid.size.z)) as usize;
                    let y = (p / grid.size.z).rem_euclid(grid.size.y) as usize;
                    let z = p.rem_euclid(grid.size.z) as usize;
                    vec![x, y, z]
                })
                .collect::<Vec<Vec<usize>>>(),
        )
        // safe to unwrap because we know that every inner vector is length 3
        .unwrap()
    }

    fn get_atom_map<'py>(
        &self,
        py: Python<'py>,
        atom_number: isize,
        density: PyReadonlyArray3<f64>,
    ) -> &'py PyArray3<f64> {
        let dims = density.dims();
        let density = if density.is_c_contiguous() {
            let g = density.shape();
            let s = &self.voxel_map.grid.size;
            assert_eq!(
                [g[0], g[1], g[2]],
                [s.x as usize, s.y as usize, s.z as usize],
                "Supplied density has different shape to the voxel map.",
            );
            // safe to unwrap from the contiguous check safe to assume 3d from argument type
            density.as_slice().unwrap()
        } else {
            panic!("Supplied reference density should be c contiguous in memory.")
        };
        PyArray1::from_iter(
            py,
            self.voxel_map
                .volume_map(atom_number)
                .into_iter()
                .zip(density)
                .map(|(weight, d)| if let Some(w) = weight { w * d } else { 0.0 }),
        )
        .reshape(dims)
        .unwrap() // safe as we know the densities are the same shape
    }
}
