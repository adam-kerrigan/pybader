use pyo3::prelude::*;

mod io;
mod voxel_map;

#[pymodule]
#[pyo3(name="_pybader")]
fn bader_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<voxel_map::PyVoxelMap>()?;
    m.add_function(wrap_pyfunction!(io::read_vasp, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_cube, m)?)?;

    Ok(())
}
