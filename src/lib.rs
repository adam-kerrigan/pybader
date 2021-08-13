use pyo3::prelude::*;
use bader::grid::Grid;
use bader::voxel_map::VoxelMap;
use bader::methods::weight;
use bader::progress::Bar;
use bader::utils::vacuum_tolerance as vacuum_tol;
use rustc_hash::FxHashMap;
use atomic_counter::{AtomicCounter, RelaxedCounter};
use crossbeam_utils::thread;


/// Recieve the charge density and optional reference density.
#[pyfunction]
fn get_voxelmap(reference: Vec<f64>, grid: [usize; 3], lattice: [[f46; 3]; 3], args: FxHashMap<String, f64>) -> PyResult<FxHashMap<u64, Vec<f64>>> {
    let vacuum_tolerance = args.get("vacuum_tolerance").map(|x| *x);
    let voxel_origin = [*args.get("vo_a").unwrap(), *args.get("vo_b").unwrap(), *args.get("vo_c").unwrap()];
    let threads = (*args.get("threads").unwrap()) as u64;
    let grid = Grid::new(grid,
                         lattice,
                         *args.get("weight_tolerance").unwrap(),
                         *args.get("maxima_tolerance").unwrap(),
                         vacuum_tolerance,
                         voxel_origin);
    let voxel_map = VoxelMap::new(grid.size.total);
    {
        let mut index: Vec<usize> = (0..grid.size.total).collect();
        // Start a thread-safe progress bar and run the main calculation
        println!("Sorting density.");
        index.sort_unstable_by(|a, b| {
                 reference[*b].partial_cmp(&reference[*a]).unwrap()
             });
        let counter = RelaxedCounter::new(0);
        let vacuum_index = vacuum_tol(&reference, &index, grid.vacuum_tolerance);
        let pbar = Bar::visible(vacuum_index as u64,
                                100,
                                String::from("Bader Partitioning: "));
        thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|_| loop {
                     let p = {
                         let i = counter.inc();
                         if i >= vacuum_index {
                             break;
                         };
                         index[i]
                     };
                     weight(p, &grid, &reference, &voxel_map);
                     pbar.tick();
                 });
            }
        }).unwrap();
    }
    {
        let mut weights = voxel_map.lock();
        weights.shrink_to_fit();
        let w = weights.copy();
        return Ok(w)
    }
}

#[pymodule]
fn bader_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_voxelmap, m)?)?;

    Ok(())
}
