#pragma once

#include <string>
#include <vector>

#include "TFUtils.hpp"

namespace tfutils
{

// --------------------------------------------------------
// Graph info
// --------------------------------------------------------
const char* TFDataTypeToString(TF_DataType dataType);

void printOpInputs(TF_Graph*, TF_Operation* op);
void printOpOutputs(TF_Graph* graph, TF_Operation* op);
void printOp(TF_Graph* graph);

} // namespace tfutils