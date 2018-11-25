#include "TFInfoUtils.hpp"

namespace tfutils
{

// --------------------------------------------------------
// Common
// --------------------------------------------------------
const char* TFDataTypeToString(TF_DataType dataType)
{
	switch (dataType)
	{
	case TF_FLOAT:
		return "TF_FLOAT";
	case TF_DOUBLE:
		return "TF_DOUBLE";
	case TF_INT32:
		return "TF_INT32";
	case TF_UINT8:
		return "TF_UINT8";
	case TF_INT16:
		return "TF_INT16";
	case TF_INT8:
		return "TF_INT8";
	case TF_STRING:
		return "TF_STRING";
	case TF_COMPLEX64:
		return "TF_COMPLEX64";
	case TF_INT64:
		return "TF_INT64";
	case TF_BOOL:
		return "TF_BOOL";
	case TF_QINT8:
		return "TF_QINT8";
	case TF_QUINT8:
		return "TF_QUINT8";
	case TF_QINT32:
		return "TF_QINT32";
	case TF_BFLOAT16:
		return "TF_BFLOAT16";
	case TF_QINT16:
		return "TF_QINT16";
	case TF_QUINT16:
		return "TF_QUINT16";
	case TF_UINT16:
		return "TF_UINT16";
	case TF_COMPLEX128:
		return "TF_COMPLEX128";
	case TF_HALF:
		return "TF_HALF";
	case TF_RESOURCE:
		return "TF_RESOURCE";
	case TF_VARIANT:
		return "TF_VARIANT";
	case TF_UINT32:
		return "TF_UINT32";
	case TF_UINT64:
		return "TF_UINT64";
	default:
		return "Unknown";
	}
}

// --------------------------------------------------------
// Graph info
// --------------------------------------------------------
void printOpInputs(TF_Graph*, TF_Operation* op)
{
	const int numInputs = TF_OperationNumInputs(op);

	std::cout << "Number of inputs: " << numInputs << std::endl;

	for (int i = 0; i < numInputs; i++)
	{
		TF_Input input = { op, i };
		TF_DataType dataType = TF_OperationInputType(input);
		std::cout << std::to_string(i) << "dataType: " << TFDataTypeToString(dataType) << std::endl;
	}
}

void printOpOutputs(TF_Graph* graph, TF_Operation* op)
{
	TF_Status* status = TF_NewStatus();
	const int numOutputs = TF_OperationNumOutputs(op);

	std::cout << "Number of outputs: " << numOutputs << std::endl;

	for (int i = 0; i < numOutputs; i++)
	{
		TF_Output output = { op, i };
		TF_DataType dataType = TF_OperationOutputType(output);
		std::cout << std::to_string(i) << "dataType: " << TFDataTypeToString(dataType) << std::endl;

		const int numDims = TF_GraphGetTensorNumDims(graph, output, status);

		if (TF_GetCode(status) != TF_OK)
		{
			std::cerr << "Error: Can't get tensor dimensionality" << std::endl;
			continue;
		}

		std::vector<std::int64_t> dims(numDims);
		TF_GraphGetTensorShape(graph, output, dims.data(), numDims, status);

		if (TF_GetCode(status) != TF_OK)
		{
			std::cerr << "Error: Can't get tensor shape" << std::endl;
			continue;
		}

		std::cout << "Number of dims: " << numDims << std::endl;
		std::cout << "dims: " << numDims <<" [";
		for (int j = 0; j < numDims; j++)
		{
			std::cout << dims[j];
			if (j < numDims - 1)
			{
				std::cout << ",";
			}
		}
		std::cout << "]" << std::endl;
	}

	TF_DeleteStatus(status);
}

void printOp(TF_Graph* graph)
{
	TF_Operation* op;
	std:size_t pos = 0;

	while ((op = TF_GraphNextOperation(graph, &pos)) != nullptr)
	{
		const char* name = TF_OperationName(op);
		const char* type = TF_OperationOpType(op);
		const char* device = TF_OperationDevice(op);

		const int numInputs = TF_OperationNumInputs(op);
		const int numOutputs = TF_OperationNumOutputs(op);

		std::cout << "pos: " << pos << " name: " << name << " type: " << type << " device: " << device << " Number of inputs: " << numInputs << " Number of outputs: " << numOutputs << std::endl;

		printOpInputs(graph, op);
		printOpOutputs(graph, op);

		std::cout << std::endl;
	}
}


// --------------------------------------------------------
// Tensor info
// --------------------------------------------------------
void printTensorInputs(TF_Graph*, TF_Operation* op)
{
	const int numInputs = TF_OperationNumInputs(op);

	for (int i = 0; i < numInputs; i++)
	{
		const TF_Input input = { op, i };
		const TF_DataType type = TF_OperationInputType(input);
		std::cout << "Input: " << i << " type: " << TFDataTypeToString(type) << std::endl;
	}
}

void printTensorOutputs(TF_Graph* graph, TF_Operation* op)
{
	const int numOutputs = TF_OperationNumOutputs(op);
	TF_Status *status = TF_NewStatus();

	for (int i = 0; i < numOutputs; i++)
	{
		const TF_Output output = { op, i };
		const TF_DataType type = TF_OperationOutputType(output);
		const int numDims = TF_GraphGetTensorNumDims(graph, output, status);
		
		if (TF_GetCode(status) != TF_OK)
		{
			std::cout << "Error : Can't get tensor dimensionality" << std::endl;
			continue;
		}

		std::vector<std::int64_t> dims(numDims);
		std::cout << "Output: " << i << " type: " << TFDataTypeToString(type);
		TF_GraphGetTensorShape(graph, output, dims.data(), numDims, status);

		if (TF_GetCode(status) != TF_OK)
		{
			std::cout << "Error: Can't get get tensor shape" << std::endl;
			continue;
		}

		std::cout << " dims: " << numDims << " [";
		for (int d = 0; d < numDims; d++)
		{
			std::cout << dims[d];
			if (d < numDims - 1)
			{
				std::cout << ", ";
			}
		}
		std::cout << "]" << std::endl;
  }

  TF_DeleteStatus(status);
}

void printTensorInfo(TF_Graph *graph, const char *layerName)
{
	std::cout << "Tensor: " << layerName;
	TF_Operation *op = TF_GraphOperationByName(graph, layerName);

	if (op == nullptr)
	{
		std::cout << "Error: Could not get " << layerName << std::endl;
		return;
	}

	const int numInputs = TF_OperationNumInputs(op);
	const int numOutputs = TF_OperationNumOutputs(op);
	std::cout << " inputs: " << numInputs << " outputs: " << numOutputs << std::endl;

	printTensorInputs(graph, op);
	printTensorOutputs(graph, op);
}

} // namespace tfutils

