#include <core/domain_geometry.h>

MSSTOKES_OPEN_NAMESPACE

template <int dim>
DomainGeometry<dim>::DomainGeometry()
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
{
  TimerOutput::Scope timing_section(
    computing_timer, "DomainGeometry - constructor with grid generation");

  upper_right_corner(0) = 1;
  upper_right_corner(1) = 1;
  if (dim == 3)
    upper_right_corner(2) = 1;

  domain_center = 0.5 * (upper_right_corner + lower_left_corner);

  GridGenerator::hyper_rectangle(triangulation,
                                 lower_left_corner,
                                 upper_right_corner,
                                 /* colorize */ true);

  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  /*
   * All dimensions up to the last are periodic (z-direction is always
   * bounded from below and from above)
   */
  for (unsigned int d = 0; d < dim - 1; ++d)
    {
      GridTools::collect_periodic_faces(triangulation,
                                        /*b_id1*/ 2 * (d + 1) - 2,
                                        /*b_id2*/ 2 * (d + 1) - 1,
                                        /*direction*/ d,
                                        periodicity_vector);
    }

  pcout << "   Number of active cells:       " << triangulation.n_active_cells()
        << std::endl;
}



template <int dim>
DomainGeometry<dim>::~DomainGeometry()
{}



template <int dim>
void
DomainGeometry<dim>::refine_global(unsigned int n_refine)
{
  TimerOutput::Scope timing_section(computing_timer,
                                    "DomainGeometry - global refinement");

  triangulation.refine_global(n_refine);

  pcout << "   Number of active cells after global refinement:       "
        << triangulation.n_active_cells() << std::endl;
}



template <int dim>
void
DomainGeometry<dim>::write_mesh_vtu()
{
  TimerOutput::Scope timing_section(computing_timer,
                                    "DomainGeometry - write mesh to disk");

  DataOut<dim> data_out;
  data_out.attach_triangulation(triangulation);

  // Add data to indicate subdomain
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    {
      subdomain(i) = triangulation.locally_owned_subdomain();
    }
  data_out.add_data_vector(subdomain, "subdomain");

  // Now build all data patches
  data_out.build_patches();

  const std::string filename_local =
    "stokes_domain." +
    Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
    ".vtu";

  std::ofstream output(filename_local.c_str());
  data_out.write_vtu(output);

  // Write a pvtu record on master process
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        filenames.emplace_back("stokes_domain." +
                               Utilities::int_to_string(i, 4) + ".vtu");

      std::string   master_file("stokes_domain.pvtu");
      std::ofstream master_output(master_file.c_str());
      data_out.write_pvtu_record(master_output, filenames);
    }
}


MSSTOKES_CLOSE_NAMESPACE
