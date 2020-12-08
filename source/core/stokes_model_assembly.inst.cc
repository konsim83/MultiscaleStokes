#include <core/stokes_model_assembly.h>

#include <core/stokes_model_assembly.tpp>

MSSTOKES_OPEN_NAMESPACE

// template instantiations
template class Assembly::Scratch::StokesPreconditioner<2>;
template class Assembly::Scratch::StokesSystem<2>;

template class Assembly::CopyData::StokesPreconditioner<2>;
template class Assembly::CopyData::StokesSystem<2>;

template class Assembly::Scratch::StokesPreconditioner<3>;
template class Assembly::Scratch::StokesSystem<3>;

template class Assembly::CopyData::StokesPreconditioner<3>;
template class Assembly::CopyData::StokesSystem<3>;

MSSTOKES_CLOSE_NAMESPACE
