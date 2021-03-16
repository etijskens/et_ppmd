/*
 *  C++ source file for module et_ppmd.corecpp
 */


// See http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html for examples on how to use pybind11.
// The example below is modified after http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html#More-on-working-with-numpy-arrays
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double force_factor(double rij2)
{
    double rm2 = 1.0/rij2;
    double rm6 = (rm2*rm2*rm2);
    return (1.0 - 2.0*rm6)*rm6*rm2*6.0;
}

void
computeForces
    ( py::array_t<double> x    // x-coordinates of atom positions, input parameter
    , py::array_t<double> y    // y-coordinates of atom positions, input parameter
    , py::array_t<int> vl      // Verlet lists of atoms, input parameter
    , py::array_t<double> fx   // x-coordinates of atom forces, output parameter
    , py::array_t<double> fy   // y-coordinates of atom forces, output parameter
    )
{
    auto buf_x = x.request()
       , buf_y = y.request()
       , buf_fx = fx.request()
       , buf_fy = fy.request()
       ;
    auto buf_vl = vl.request();
 // Check array dimensions
    if( buf_x.ndim != 1 )
        throw std::runtime_error("Parameter x must be 1-dimensional");
    if( buf_y.ndim != 1 )
        throw std::runtime_error("Parameter y must be 1-dimensional");
    if( buf_vl.ndim != 2 )
        throw std::runtime_error("Parameter fx must be 1-dimensional");
    if( buf_fx.ndim != 1 )
        throw std::runtime_error("Parameter fx must be 1-dimensional");
    if( buf_fy.ndim != 1 )
        throw std::runtime_error("Parameter fy must be 1-dimensional");
 // Check array shapes
    std::size_t n_atoms = buf_x.shape[0];
    if( (buf_y .shape[0] != n_atoms)
     || (buf_vl.shape[0] <  n_atoms)
     || (buf_fx.shape[0] != n_atoms)
     || (buf_fy.shape[0] != n_atoms)
      ) {
        throw std::runtime_error("Input shapes don't match.");
    }
    std::size_t max_neighbours = buf_vl.shape[1];
 // because the Numpy arrays are mutable by default, py::array_t is mutable too.
 // Below we declare the raw C++ arrays for x, y and vl as const to make their intent clear.
    double const *ptrx = static_cast<double const *>(buf_x.ptr);
    double const *ptry = static_cast<double const *>(buf_y.ptr);
    int    const *ptrvl= static_cast<int    const *>(buf_vl.ptr);
    double       *ptrfx = static_cast<double       *>(buf_fx.ptr);
    double       *ptrfy = static_cast<double       *>(buf_fy.ptr);

 // Zero the forces
    for (std::size_t i=0; i<n_atoms; ++i ) {
        ptrfx[i] = 0.0;
    }
    for (std::size_t i=0; i<n_atoms; ++i ) {
        ptrfy[i] = 0.0;
    }
 // Compute the forces
    int const* vli = ptrvl;
    for (std::size_t i=0; i<n_atoms; ++i) {
        int n_neighbours = vli[0];
        vli++;
        for (std::size_t nb=0; nb<n_neighbours; ++nb, ++vli) {
            std::size_t j = vli[nb];
            double xij = ptrx[j] - ptrx[i];
            double yij = ptry[j] - ptry[i];
            double rij2 = xij*xij + yij*yij;
            double ff = force_factor(rij2);
            double fx = ff*xij;
            double fy = ff*yij;
            ptrfx[i] += fx;
            ptrfy[i] += fy;
            ptrfx[j] -= fx;
            ptrfy[j] -= fy;
        }
    }
}


PYBIND11_MODULE(corecpp, m)
{// optional module doc-string
    m.doc() = "pybind11 corecpp plugin"; // optional module docstring
 // list the functions you want to expose:
 // m.def("exposed_name", function_pointer, "doc-string for the exposed function");
    m.def("computeForces", &computeForces, "Compute the Lennard-Jones interaction forces.");
}
