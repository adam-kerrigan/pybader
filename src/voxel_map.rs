use bader::analysis::assign_maxima;
use bader::analysis::calculate_bader_volume_radius;
use bader::atoms::{Atoms, Lattice};
use bader::grid::Grid;
use bader::methods::{maxima_finder, weight};
use bader::progress::Bar;
use bader::utils::vacuum_index;
use bader::voxel_map::AtomVoxelMap;
use bader::voxel_map::BlockingVoxelMap;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Return a "Voxel Map" to python.
///
/// Build a map of each voxel in the supplied reference density to cartesian
/// positions within the lattice supplied.
#[pyfunction]
pub fn get_voxelmap<'py>(
    py: Python<'py>,
    reference: PyReadonlyArray1<f64>,
    grid: [usize; 3],
    lattice: [[f64; 3]; 3],
    positions: Vec<[f64; 3]>,
    voxel_origin: [f64; 3],
    visible_bar: bool,
    weight_tolerance: f64,
    maximum_distance: f64,
    threads: usize,
    vacuum_tolerance: Option<f64>,
) -> PyResult<(Vec<isize>, &'py PyArray1<isize>, Vec<Vec<f64>>)> {
    // This can fail if the inputted density is no contiguous
    // Not a problem if density supplied is formated correctly
    let reference = match reference.as_slice() {
        Ok(r) => r,
        _ => panic!("Supplied reference density should be contiguous in memory."),
    };
    let atoms = Atoms::new(Lattice::new(lattice), positions, String::new());
    let voxel_map = BlockingVoxelMap::new(grid, lattice, voxel_origin);
    let mut index: Vec<usize> = (0..voxel_map.grid.size.total).collect();
    // This can be unwrapped as we know there are no NaN values.
    index.sort_unstable_by(|a, b| reference[*b].partial_cmp(&reference[*a]).unwrap());
    // remove from the indices any voxel that is below the vacuum limit
    let vac_index = match vacuum_index(&reference, &index, vacuum_tolerance) {
        Ok(i) => i,
        _ => panic!("Vacuum tolerance is higher than maximum value of density."),
    };
    index.truncate(vac_index);
    // find the maxima in the system and store them whilst removing them from
    // the index list
    let pbar = match visible_bar {
        true => Bar::visible(index.len() as u64, 100, String::from("Maxima Finding: ")),
        false => Bar::new(index.len() as u64, 100, String::from("Maxima Finding: ")),
    };
    let bader_maxima = maxima_finder(&index, &reference, &voxel_map, threads, pbar).unwrap();
    // Assign the maxima to atoms
    let pbar = match visible_bar {
        true => Bar::visible(
            bader_maxima.len() as u64,
            100,
            String::from("Assigning to Atoms: "),
        ),
        false => Bar::new(
            bader_maxima.len() as u64,
            100,
            String::from("Assigning to Atoms: "),
        ),
    };
    let atom_map = match assign_maxima(
        &bader_maxima,
        &atoms,
        &voxel_map.grid,
        &maximum_distance,
        threads,
        pbar,
    ) {
        Ok(am) => am,
        // I need to start properly handling errors in bader-rs now as it would be nice to
        // pass the error up from the function as that will be better.
        _ => panic!("Cannot assign maxima to atoms."),
    };
    // input the maxima as atoms into the voxel map
    bader_maxima.iter().enumerate().for_each(|(i, maxima)| {
        voxel_map.maxima_store(*maxima, atom_map[i] as isize);
    });
    let pbar = match visible_bar {
        true => Bar::visible(
            bader_maxima.len() as u64,
            100,
            String::from("Assigning to Atoms: "),
        ),
        false => Bar::new(
            bader_maxima.len() as u64,
            100,
            String::from("Assigning to Atoms: "),
        ),
    };
    // calculate the weights and store the saddles
    let saddles = weight(
        &reference,
        &voxel_map,
        &index,
        pbar,
        threads,
        weight_tolerance,
    );
    let (atom_voxel_map, weight_map, _) = voxel_map.into_inner();
    Ok((
        saddles.keys().copied().collect(),
        atom_voxel_map.into_pyarray(py),
        weight_map.into_iter().map(|v| v.into_vec()).collect(),
    ))
}

#[pyfunction]
pub fn calculate_volume_and_radius<'py>(
    py: Python<'py>,
    voxel_map: PyReadonlyArray1<isize>,
    weight_map: Vec<Vec<f64>>,
    lattice: [[f64; 3]; 3],
    grid: [usize; 3],
    voxel_origin: [f64; 3],
    positions: Vec<[f64; 3]>,
    density: PyReadonlyArray1<f64>,
    threads: usize,
    visible_bar: bool,
) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
    let density = match density.as_slice() {
        Ok(r) => r,
        _ => panic!("Supplied reference density should be contiguous in memory."),
    };
    let grid = Grid::new(grid, lattice, voxel_origin);
    let voxel_map = match voxel_map.as_slice() {
        Ok(r) => r.to_vec(),
        _ => panic!("Supplied reference density should be contiguous in memory."),
    };
    let voxel_map = AtomVoxelMap::new(
        voxel_map,
        weight_map
            .into_iter()
            .map(|v| v.into_boxed_slice())
            .collect(),
        grid,
    );
    let atoms = Atoms::new(Lattice::new(lattice), positions, String::new());
    let pbar = match visible_bar {
        true => Bar::visible(
            density.len() as u64,
            100,
            String::from("Calculating Volumes: "),
        ),
        false => Bar::new(
            density.len() as u64,
            100,
            String::from("Calculating Volumes: "),
        ),
    };
    let (bv, br) = calculate_bader_volume_radius(density, &voxel_map, &atoms, threads, pbar);
    (bv.into_pyarray(py), br.into_pyarray(py))
}
