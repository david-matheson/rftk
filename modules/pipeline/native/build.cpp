#include "AllSamplesStep.h"
#include "BootstrapSamplesStep.h"
#include "SetBufferStep.h"
#include "SliceBufferStep.h"

template class BootstrapSamplesStep< float, int >;
template class SetBufferStep< VectorBufferTemplate<float> >;
template class SetBufferStep< VectorBufferTemplate<double> >;
template class SetBufferStep< VectorBufferTemplate<int> >;
template class SetBufferStep< VectorBufferTemplate<long long> >;

template class SetBufferStep< MatrixBufferTemplate<float> >;
template class SetBufferStep< MatrixBufferTemplate<double> >;
template class SetBufferStep< MatrixBufferTemplate<int> >;
template class SetBufferStep< MatrixBufferTemplate<long long> >;