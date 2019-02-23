#pragma once

#include <string>
#include <vector>

#include "TFUtils.hpp"

namespace tfutils
{

    // --------------------------------------------------------
    // Common
    // --------------------------------------------------------
    const char* TFDataTypeToString(TF_DataType dataType);
    
    // --------------------------------------------------------
    // Graph info
    // --------------------------------------------------------
    void printOpInputs(TF_Graph*, TF_Operation* op);
    void printOpOutputs(TF_Graph* graph, TF_Operation* op);
    void printOp(TF_Graph* graph);
    
    // --------------------------------------------------------
    // Tensor info
    // --------------------------------------------------------
    void printTensorInputs(TF_Graph*, TF_Operation* op);
    void printTensorOutputs(TF_Graph* graph, TF_Operation* op);
    void printTensorInfo(TF_Graph *graph, const char *layerName);

} // namespace tfutils