###############################################################################
###############################################################################
#
# Include directory for sources
#
include_directories(${MsStokes_INCLUDE_DIR})
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Name all sources
#
#file(GLOB_RECURSE MsStokes_TARGET_LIB_SRC  "*.cc") # source files
#set(MsStokes_TARGET_LIB_SRC ${MsStokes_TARGET_LIB_SRC})
set(MsStokes_TARGET_LIB_SRC
	base/utilities.cc
	model_data/physical_constants.cc
	model_data/stokes_model_data.inst.cc
	model_data/stokes_model_parameters.cc
	model_data/core_model_data.inst.cc
	core/domain_geometry.inst.cc
	multiscale_basis/divergence_stabilized_basis.inst.cc
	core/multiscale_stokes_model.inst.cc
	core/stokes_model.inst.cc
	core/stokes_model_assembly.inst.cc
	postprocessor/postprocessor.inst.cc)

print_all_args (
  ${MsStokes_TARGET_LIB_SRC}
 )
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Compile and link the sources as SHARED
#
add_library (MsStokes_LIBRARY SHARED ${MsStokes_TARGET_LIB_SRC})
DEAL_II_SETUP_TARGET(MsStokes_LIBRARY)

add_executable(Stokes "stokes.cxx")
DEAL_II_SETUP_TARGET(Stokes)
TARGET_LINK_LIBRARIES(Stokes MsStokes_LIBRARY)

add_executable(MsStokes "multi_stokes.cxx")
DEAL_II_SETUP_TARGET(MsStokes)
TARGET_LINK_LIBRARIES(MsStokes MsStokes_LIBRARY)

#
# Install into the DESTINATION provided by CMAKE_INSTALL_PREFIX
#
#install (TARGETS ${MsStokes_LIBRARY} DESTINATION ${CMAKE_INSTALL_PREFIX})
###############################################################################
###############################################################################
