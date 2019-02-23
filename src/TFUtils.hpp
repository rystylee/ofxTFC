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
#include <vector>

#include "c_api.h"

namespace tfutils
{

    // --------------------------------------------------------
    // Graph
    // --------------------------------------------------------
    TF_Graph* loadGraphDef(const char* graphName);
    
    //TF_Output loadOperation(TF_Graph* graph, const char* opName);
    const std::vector<TF_Output> loadOperations(TF_Graph* graph, const char* opName);
    
    // --------------------------------------------------------
    // Session
    // --------------------------------------------------------
    TF_Session* createSession(TF_Graph* graph);
    
    bool runSession(TF_Graph* graph,
                    const TF_Output* inputOps, TF_Tensor* const* inputTensors, std::size_t numInputs,
                    const TF_Output* outputOps, TF_Tensor** outputTensors, std::size_t numOutputs);
    
    bool runSession(TF_Graph* graph,
                    const std::vector<TF_Output>& inputOps, const std::vector<TF_Tensor*>& inputTensors,
                    const std::vector<TF_Output>& outputOps, std::vector<TF_Tensor*>& outputTensors);
    
    bool runSession(TF_Session* sess,
                    const TF_Output* inputOps, TF_Tensor* const* inputTensors, std::size_t numInputs,
                    const TF_Output* outputOps, TF_Tensor** outputTensors, std::size_t numOutputs);
    
    bool runSession(TF_Session* sess,
                    const std::vector<TF_Output>& inputOps, const std::vector<TF_Tensor*>& inputTensors,
                    const std::vector<TF_Output>& outputOps, std::vector<TF_Tensor*>& outputTensors);
    
    // --------------------------------------------------------
    // Tensor
    // --------------------------------------------------------
    TF_Tensor* createTensor(TF_DataType dataType,
                            const std::int64_t* dims, std::size_t numDims,
                            const void* data, std::size_t len);
    
    template<typename T>
    TF_Tensor* createTensor(TF_DataType dataType,
                            const std::vector<std::int64_t>& dims,
                            const std::vector<T>& data)
    {
        return createTensor(dataType,
                            dims.data(), dims.size(),
                            data.data(), data.size() * sizeof(T));
    }
    
    void deleteTensor(TF_Tensor* tensor);
    void deleteTensors(const std::vector<TF_Tensor*>& tensor);
    
    template<typename T>
    std::vector<T> tensorData(const TF_Tensor* tensor)
    {
        const auto data = static_cast<T*>(TF_TensorData(tensor));
        if (data == nullptr)
        {
            std::cerr << "Error: tensor is null" << std::endl;
            return {};
        }
    
        return { data, data + (TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor))) };
    }
    
    template<typename T>
    std::vector<std::vector<T>> tensorData(const std::vector<TF_Tensor*>& tensors)
    {
        std::vector<std::vector<T>> data;
        data.reserve(tensors.size());
        for (const auto& e : tensors)
        {
            data.push_back(tensorData<T>(e));
        }
    
        return data;
    }

} // namespace tfutils
