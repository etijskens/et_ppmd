/*
 *  C++ source file for module et_ppmd.corecpp
 */

#include <iostream>

// VERBOSE is for debugging purposes
// #define VERBOSE

// See http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html for examples on how to use pybind11.
// The example below is modified after http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html#More-on-working-with-numpy-arrays
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

//------------------------------------------------------------------------------
double force_factor(double rij2)
 // Compute the force factor. Multiply with the interatomic vector to obtain the
 // force vector.
{
    double rm2 = 1.0/rij2;
    double rm6 = (rm2*rm2*rm2);
    return (1.0 - 2.0*rm6)*rm6*rm2*6.0;
}

//------------------------------------------------------------------------------
void
computeForces                    // input parameters:
    ( py::array_t<double> x      // x-coordinates of atom positions
    , py::array_t<double> y      // y-coordinates of atom positions
    , py::array_t<int>    vlSize // array with the size of the Verlet lists
    , py::array_t<int>    vlLsts // the linearized Verlet lists
                                 // output parameters:
    , py::array_t<double> fx     // x-coordinates of atom forces
    , py::array_t<double> fy     // y-coordinates of atom forces
    )
 // Compute the forces from the atom positions (x,y) and the Verlet list vl.
 // See the VerletList class in et_ppmd/verlet.py for an explanation of the
 // data structure (a 2D integer array.
{
 #ifdef VERBOSE
    std::cout << "entering corecpp" << std::endl;
 #endif
    auto buf_x = x.request()
       , buf_y = y.request()
       , buf_fx = fx.request()
       , buf_fy = fy.request()
       ;
    auto buf_vlsz = vlSize.request();
    auto buf_vlst = vlLsts.request();
 // Check array dimensions
    if( buf_x.ndim != 1 )
        throw std::runtime_error("Parameter x must be 1-dimensional");
    if( buf_y.ndim != 1 )
        throw std::runtime_error("Parameter y must be 1-dimensional");
    if( buf_vlsz.ndim != 1 )
        throw std::runtime_error("Parameter vlSize must be 1-dimensional");
    if( buf_vlst.ndim != 1 )
        throw std::runtime_error("Parameter vlLsts must be 1-dimensional");
    if( buf_fx.ndim != 1 )
        throw std::runtime_error("Parameter fx must be 1-dimensional");
    if( buf_fy.ndim != 1 )
        throw std::runtime_error("Parameter fy must be 1-dimensional");
 // Check array shapes
    std::size_t n_atoms = buf_x.shape[0];
    if( (buf_y   .shape[0] != n_atoms)
     || (buf_vlsz.shape[0] != n_atoms)
     || (buf_fx  .shape[0] != n_atoms)
     || (buf_fy  .shape[0] != n_atoms)
      ) {
        throw std::runtime_error("Parameter shapes don't match.");
    }
//    std::size_t max_neighbours = buf_vl.shape[1] - 1; // mind the minus 1
  #ifdef VERBOSE
    std::cout << "corecpp: n_atoms=" << n_atoms << " max_neighbours=" << max_neighbours << std::endl;
  #endif

 // because the Numpy arrays are mutable by default, py::array_t is mutable too.
 // Below we declare the raw C++ arrays for x, y and vl as const to make their intent clear.
    double const *ptr_x    = static_cast<double const *>(buf_x.ptr);
    double const *ptr_y    = static_cast<double const *>(buf_y.ptr);
    int    const *ptr_vlsz = static_cast<int    const *>(buf_vlsz.ptr);
    int    const *ptr_vlst = static_cast<int    const *>(buf_vlst.ptr);
    double       *ptr_fx   = static_cast<double       *>(buf_fx.ptr);
    double       *ptr_fy   = static_cast<double       *>(buf_fy.ptr);

 // Zero the forces
 // one may argue whether this needs to be done here or in python using numpy
    for (std::size_t i=0; i<n_atoms; ++i ) {
        ptr_fx[i] = 0.0;
    }
    for (std::size_t i=0; i<n_atoms; ++i ) {
        ptr_fy[i] = 0.0;
    }
 // Compute the interatomic forces and add them to fx and fy
 // loop over all atoms
    std::size_t nb0 = 0; // the starting index of the Verlet list
    for (std::size_t i=0; i<n_atoms; ++i) {
     // the length of the Verlet list of atom i
        int n_neighbours_i = ptr_vlsz[i];
     // the verlet list of atom i
        int const * vli = &ptr_vlst[nb0];
      #ifdef VERBOSE
        std::cout << "corecpp: i=" << i << " nn=" << n_neighbours << std::endl;
      #endif
        for (std::size_t nb=0; nb<n_neighbours_i; ++nb) {
            std::size_t j = vli[nb];
          #ifdef VERBOSE
            std::cout << "corecpp: nb=" << nb << " j=" << j << std::endl;
          #endif
            double xij = ptr_x[j] - ptr_x[i];
            double yij = ptr_y[j] - ptr_y[i];
            double rij2 = xij*xij + yij*yij;
            double ff = force_factor(rij2);
            double fx = ff*xij;
            double fy = ff*yij;
            ptr_fx[i] += fx;
            ptr_fy[i] += fy;
            ptr_fx[j] -= fx;
            ptr_fy[j] -= fy;
        }
     // move to the next Verlet list
        nb0 += n_neighbours_i;
    }
  #ifdef VERBOSE
    std::cout << " corecpp: ax=[ " ;
    for (std::size_t i = 0; i<n_atoms; ++i){
        std::cout << ptr_fx[i] << ' ';
    }
    std::cout << ']' << std::endl;
    std::cout << "exiting corecpp" << std::endl;
  #endif
}

//------------------------------------------------------------------------------
void
velocity_verlet_12
    ( double              dt            // timestep
    , py::array_t<double> rx            // x-coordinates of atom positions
    , py::array_t<double> ry            // y-coordinates of atom positions
    , py::array_t<double> vx            // x-coordinates of atom velocities
    , py::array_t<double> vy            // y-coordinates of atom velocities
    , py::array_t<double> ax            // x-coordinates of atom accelerations
    , py::array_t<double> ay            // y-coordinates of atom accelerations
    , py::array_t<double> vx_midstep    // x-coordinates of midstep velocities
    , py::array_t<double> vy_midstep    // y-coordinates of midstep velocities
    )
 // velocity verlet algorithm step 1 and 2
{
    auto buf_rx = rx.request()
       , buf_ry = ry.request()
       , buf_vx = vx.request()
       , buf_vy = vy.request()
       , buf_ax = ax.request()
       , buf_ay = ay.request()
       , buf_vx_midstep = vx_midstep.request()
       , buf_vy_midstep = vy_midstep.request()
       ;

 // because the Numpy arrays are mutable by default, py::array_t is mutable too.
 // Below we declare the raw C++ arrays for x, y and vl as const to make their intent clear.
    double       *ptr_rx = static_cast<double       *>(buf_rx.ptr);
    double       *ptr_ry = static_cast<double       *>(buf_ry.ptr);
    double const *ptr_vx = static_cast<double const *>(buf_vx.ptr);
    double const *ptr_vy = static_cast<double const *>(buf_vy.ptr);
    double const *ptr_ax = static_cast<double const *>(buf_ax.ptr);
    double const *ptr_ay = static_cast<double const *>(buf_ay.ptr);
    double       *ptr_vx_midstep    = static_cast<double *>(buf_vx_midstep.ptr);
    double       *ptr_vy_midstep    = static_cast<double *>(buf_vy_midstep.ptr);

    std::size_t n_atoms = buf_rx.shape[0];

 // Step 1: compute velocities at midstep (t+dt/2) using the current accelerations:
 //    self.vx_midstep = self.vx + (0.5*dt)*self.ax
 //    self.vy_midstep = self.vy + (0.5*dt)*self.ay
 // Step 2: compute positions at next step (t+dt) using the midstep velocities:
 //     self.rx += self.vx_midstep*dt
 //     self.ry += self.vy_midstep*dt
 // first all work on x components, then on y-components

    double halfstep = 0.5*dt;

    for (std::size_t i=0; i<n_atoms; ++i) {
        ptr_vx_midstep[i] = ptr_vx[i] + halfstep*ptr_ax[i];
        ptr_rx[i] += ptr_vx_midstep[i] * dt;
    }

    for (std::size_t i=0; i<n_atoms; ++i) {
        ptr_vy_midstep[i] = ptr_vy[i] + halfstep*ptr_ay[i];
        ptr_ry[i] += ptr_vy_midstep[i] * dt;
    }
}
//------------------------------------------------------------------------------
void
velocity_verlet_4
    ( double              dt            // timestep
    , py::array_t<double> rx            // x-coordinates of atom positions
    , py::array_t<double> ry            // y-coordinates of atom positions
    , py::array_t<double> vx            // x-coordinates of atom velocities
    , py::array_t<double> vy            // y-coordinates of atom velocities
    , py::array_t<double> ax            // x-coordinates of atom accelerations
    , py::array_t<double> ay            // y-coordinates of atom accelerations
    , py::array_t<double> vx_midstep    // x-coordinates of midstep velocities
    , py::array_t<double> vy_midstep    // y-coordinates of midstep velocities
    )
 // velocity verlet algorithm step 4
{
    auto buf_rx = rx.request()
       , buf_ry = ry.request()
       , buf_vx = vx.request()
       , buf_vy = vy.request()
       , buf_ax = ax.request()
       , buf_ay = ay.request()
       , buf_vx_midstep = vx_midstep.request()
       , buf_vy_midstep = vy_midstep.request()
       ;

 // because the Numpy arrays are mutable by default, py::array_t is mutable too.
 // Below we declare the raw C++ arrays for x, y and vl as const to make their intent clear.
    double const *ptr_rx = static_cast<double const *>(buf_rx.ptr);
    double const *ptr_ry = static_cast<double const *>(buf_ry.ptr);
    double       *ptr_vx = static_cast<double       *>(buf_vx.ptr);
    double       *ptr_vy = static_cast<double       *>(buf_vy.ptr);
    double const *ptr_ax = static_cast<double const *>(buf_ax.ptr);
    double const *ptr_ay = static_cast<double const *>(buf_ay.ptr);
    double const *ptr_vx_midstep    = static_cast<double const *>(buf_vx_midstep.ptr);
    double const *ptr_vy_midstep    = static_cast<double const *>(buf_vy_midstep.ptr);

    std::size_t n_atoms = buf_rx.shape[0];

 // Step 4: compute velocities at next step (t+dt)
 //		self.vx = self.vx_midstep + self.ax * (0.5*dt)
 //		self.vy = self.vy_midstep + self.ay * (0.5*dt)

    double halfstep = 0.5*dt;

    for (std::size_t i=0; i<n_atoms; ++i) {
        ptr_vx[i] = ptr_vx_midstep[i] + halfstep*ptr_ax[i];
    }

    for (std::size_t i=0; i<n_atoms; ++i) {
        ptr_vy[i] = ptr_vy_midstep[i] + halfstep*ptr_ay[i];
    }
}
//------------------------------------------------------------------------------

PYBIND11_MODULE(corecpp, m)
{// optional module doc-string
    m.doc() = "pybind11 corecpp plugin"; // optional module docstring
 // list the functions you want to expose:
 // m.def("exposed_name", function_pointer, "doc-string for the exposed function");
    m.def("computeForces", &computeForces, "Compute the Lennard-Jones interaction forces.");
    m.def("velocity_verlet_12", &velocity_verlet_12, "velocity verlet steps 1 and 2.");
    m.def("velocity_verlet_4" , &velocity_verlet_4 , "velocity verlet step 4.");
}
