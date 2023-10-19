use bader::io::{cube, vasp, FileFormat};
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;


#[pyfunction]
pub fn read_vasp(
    py: Python<'_>,
    filename: String,
) -> PyResult<(
    Vec<&PyArray3<f64>>,
    &PyArray2<f64>,
    &PyArray2<f64>,
)> {
    let v = vasp::Vasp {};
    let (_, grid, atoms, density) = match v.read(filename.clone()) {
        Ok(o) => o,
        Err(_) => panic!("Cannot read file: {}", filename),
    };
    // This is very messy and annoying but I guess people would assume to read their lattice and
    // not flip a and c
    let density = density
        .into_iter()
        .map(|v| {
            let mut d = Array3::from_shape_vec((grid[0], grid[1], grid[2]), v).unwrap();
            d.swap_axes(0, 2);
            PyArray1::from_vec(py, d.into_iter().collect::<Vec<f64>>())
                .reshape([grid[2], grid[1], grid[0]])
                .unwrap()
        })
        .collect();
    let lattice = atoms
        .lattice
        .to_cartesian
        .iter()
        .rev()
        .map(|v| v.iter().rev().copied().collect())
        .collect::<Vec<Vec<f64>>>();
    let atoms = atoms
        .positions
        .iter()
        .map(|v| v.iter().rev().copied().collect())
        .collect::<Vec<Vec<f64>>>();
    Ok((
        density,
        PyArray2::from_vec2(py, &lattice).unwrap(),
        PyArray2::from_vec2(py, &atoms).unwrap(),
    ))
}

#[pyfunction]
pub fn read_cube(
    py: Python<'_>,
    filename: String,
) -> PyResult<(
    Vec<&PyArray3<f64>>,
    &PyArray2<f64>,
    &PyArray2<f64>,
)> {
    let c = cube::Cube {};
    let (_, grid, atoms, density) = match c.read(filename.clone()) {
        Ok(o) => o,
        Err(_) => panic!("Cannot read file: {}", filename),
    };
    let density = density
        .into_iter()
        .map(|v| {
            v.into_pyarray(py)
                .reshape_with_order(grid, numpy::npyffi::NPY_ORDER::NPY_CORDER)
                .unwrap()
        })
        .collect();
    let lattice = atoms
        .lattice
        .to_cartesian
        .iter()
        .map(|v| v.to_vec())
        .collect::<Vec<Vec<f64>>>();
    let atoms = atoms
        .positions
        .iter()
        .map(|v| v.to_vec())
        .collect::<Vec<Vec<f64>>>();
    Ok((
        density,
        PyArray2::from_vec2(py, &lattice).unwrap(),
        PyArray2::from_vec2(py, &atoms).unwrap(),
    ))
}
