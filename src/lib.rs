use pyo3::prelude::*;

mod voxel_map;

#[pymodule]
fn bader_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(voxel_map::get_voxelmap, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_map::calculate_volume_and_radius, m)?)?;

    Ok(())
}
