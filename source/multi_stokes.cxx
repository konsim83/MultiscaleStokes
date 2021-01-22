// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/utilities.h>

#include <core/multiscale_stokes_model.h>

// C++ STL
#include <fstream>
#include <iostream>

// AquaPlanet


using namespace dealii;


int
main(int argc, char *argv[])
{
  // Very simple way of input handling.
  if (argc < 2)
    {
      std::cout << "You must provide an input file \"-p <filename>\""
                << std::endl;
      exit(1);
    }

  std::string input_file = "";

  std::list<std::string> args;
  for (int i = 1; i < argc; ++i)
    {
      args.push_back(argv[i]);
    }

  while (args.size())
    {
      if (args.front() == std::string("-p"))
        {
          if (args.size() == 1) /* This is not robust. */
            {
              std::cerr << "Error: flag '-p' must be followed by the "
                        << "name of a parameter file." << std::endl;
              exit(1);
            }
          else
            {
              args.pop_front();
              input_file = args.front();
              args.pop_front();
            }
        }
      else
        {
          std::cerr << "Unknown command line option: " << args.front()
                    << std::endl;
          exit(1);
        }
    } // end while

#ifdef DEBUG
#  ifdef LIMIT_THREADS_FOR_DEBUG
  MultithreadInfo::set_thread_limit(1);
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, /* max_threads */ 1);
#  else
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, dealii::numbers::invalid_unsigned_int);
#  endif
#else
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, dealii::numbers::invalid_unsigned_int);
#endif


  try
    {
      MsStokes::CoreModelData::Parameters parameters(input_file);

      if (parameters.hello_from_cluster)
        {
          char processor_name[MPI_MAX_PROCESSOR_NAME];
          int  name_len;
          MPI_Get_processor_name(processor_name, &name_len);

          std::string proc_name(processor_name, name_len);

          std::cout << "Hello from   " << proc_name << "   Rank:   "
                    << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                    << "   out of   "
                    << dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                    << "   | cores = " << dealii::MultithreadInfo::n_cores()
                    << "   | threads = " << dealii::MultithreadInfo::n_threads()
                    << std::endl
                    << std::endl;
        }

      dealii::deallog.depth_console(parameters.solver_diagnostics_print_level);

      if (parameters.space_dimension == 2)
        {
          MsStokes::MultiscaleStokesModel<2> stokes_problem(parameters);
          stokes_problem.run();
        }
      else if (parameters.space_dimension == 3)
        {
          MsStokes::MultiscaleStokesModel<3> stokes_problem(parameters);
          stokes_problem.run();
        }
      else
        {
          std::cerr << "Invalid space dimension." << std::endl;
          exit(1);
        }
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;

      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;

      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
