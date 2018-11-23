#pragma once

#if defined(_MSC_VER)
#  if !defined(COMPILER_MSVC)
#    define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#  endif
#  pragma warning(push)
#  pragma warning(disable : 4190)
#endif

#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "c_api.h"

namespace tfutils
{

// --------------------------------------------------------
// Graph
// --------------------------------------------------------
TF_Graph* loadGraphDef(const char* graphName);

TF_Output loadOperation(TF_Graph* graph, const char* opName);

// --------------------------------------------------------
// Session
// --------------------------------------------------------
TF_Session* createSession(TF_Graph* graph);

bool runSession(TF_Graph* graph,
				const TF_Output* inputOps, TF_Tensor* const* inputTensors, std::size_t numInputs,
				const TF_Output* outputOps, TF_Tensor** outputTensors, std::size_t numOutputs);

bool runSession(TF_Session* sess,
				const TF_Output* inputOps, TF_Tensor* const* inputTensors, std::size_t numInputs,
				const TF_Output* outputOps, TF_Tensor** outputTensors, std::size_t numOutputs);

// --------------------------------------------------------
// Tensor
// --------------------------------------------------------
TF_Tensor* createTensor(TF_DataType dataType,
						const std::int64_t* dims, std::size_t numDims,
						const void* data, std::size_t len);

void DeleteTensor(TF_Tensor* tensor);

} // namespace tfutils
