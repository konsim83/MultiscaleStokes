#pragma once

// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>


// C++ STL
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


// MsStokes
#include <base/config.h>
#include <model_data/stokes_model_data.h>


MSSTOKES_OPEN_NAMESPACE


/*!
 * Base class to handle mesh for aqua planet. The mesh
 * is a 3D spherical shell. Derived classes will implement different model
 * details.
 */
template <int dim>
class DomainGeometry
{
public:
  /*!
   * Constructor of mesh handler for spherical shell.
   */
  DomainGeometry();
  ~DomainGeometry();

protected:
  void
  write_mesh_vtu();

  void
  refine_global(unsigned int n_refine);

  MPI_Comm mpi_communicator;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

  parallel::distributed::Triangulation<dim> triangulation;

  Point<dim> lower_left_corner, upper_right_corner;
};


// Extern template instantiations
extern template class DomainGeometry<2>;
extern template class DomainGeometry<3>;

MSSTOKES_CLOSE_NAMESPACE
