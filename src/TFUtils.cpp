#include "TFUtils.hpp"

namespace tfutils
{

namespace
{

static void freeBuffer(void* data, size_t length)
{
	std::free(data);
}

static TF_Buffer* readBufferFromFile(const char* fileName)
{
	const auto f = std::fopen(fileName, "rb");
	if (f == nullptr)
	{
		std::cerr << "Error: " << fileName << " doesn't exits" << std::endl;
		return nullptr;
	}

	std::fseek(f, 0, SEEK_END);
	const auto fsize = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);

	if (fsize < 1)
	{
		std::cerr << "Error: " << fileName << " is empty" << std::endl;
		std::fclose(f);
		return nullptr;
	}
	
	const auto data = std::malloc(fsize);
	std::fread(data, fsize, 1, f);
	std::fclose(f);

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = tfutils::freeBuffer;

	return buf;
}

} // namespace tfutils::

// --------------------------------------------------------
// Graph
// --------------------------------------------------------
TF_Graph* loadGraphDef(const char* graphName)
{
	TF_Buffer* buffer = readBufferFromFile(graphName);
	if (buffer == nullptr)
	{
		std::cerr << "Error: Can't read buffer from file" << std::endl;
		return nullptr;
	}

	TF_Graph* graph = TF_NewGraph();
	TF_Status* status = TF_NewStatus();
	TF_ImportGraphDefOptions* options = TF_NewImportGraphDefOptions();

	TF_GraphImportGraphDef(graph, buffer, options, status);
	TF_DeleteImportGraphDefOptions(options);
	TF_DeleteBuffer(buffer);

	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't load GraphDef" << std::endl;
		TF_DeleteGraph(graph);
		graph = nullptr;
	}

	TF_DeleteStatus(status);

	std::cout << "Successfully load GraphDef" << std::endl;
	
	return graph;
}

TF_Output loadOperation(TF_Graph* graph, const char* opName)
{
	TF_Output op = { TF_GraphOperationByName(graph, opName), 0 };
	if (op.oper == nullptr)
	{
		std::cerr << "Error: Can't initialize inputOperation" << std::endl;
		return op;
	}

	std::cout << "Successfully load GraphOperation" << std::endl;

	return op;
}

// --------------------------------------------------------
// Session
// --------------------------------------------------------
TF_Session* createSession(TF_Graph* graph)
{
	if (graph == nullptr)
	{
		std::cerr << "Error: TF_Graph is nullptr" << std::endl;
		return nullptr;
	}

	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, options, status);

	TF_DeleteSessionOptions(options);

	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't create session" << std::endl;
		TF_DeleteStatus(status);
		sess = nullptr;
	}

	TF_DeleteStatus(status);

	return sess;
}

bool runSession(
	TF_Graph* graph,
	const TF_Output* inputOps, TF_Tensor* const* inputTensors, std::size_t numInputs,
	const TF_Output* outputOps, TF_Tensor** outputTensors, std::size_t numOutputs)
{
	if (graph == nullptr || inputOps == nullptr || inputTensors == nullptr || outputOps == nullptr || outputTensors == nullptr)
	{
		return false;
	}

	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* ops = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, ops, status);
	TF_DeleteSessionOptions(ops);

	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't create session" << std::endl;
		TF_DeleteStatus(status);
		return false;
	}

	TF_SessionRun(
		sess,
		nullptr, // Run options
		inputOps, inputTensors, static_cast<int>(numInputs),
		outputOps, outputTensors, static_cast<int>(numOutputs),
		nullptr, 0, // Target oprations, number of targets
		nullptr, // Run metadata
		status // Output status
	);

	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't run session" << std::endl;
		TF_CloseSession(sess, status);
		TF_DeleteSession(sess, status);
		TF_DeleteStatus(status);
		return false;
	}

	TF_CloseSession(sess, status);
	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't close session" << std::endl;
		TF_CloseSession(sess, status);
		TF_DeleteSession(sess, status);
		TF_DeleteStatus(status);
		return false;
	}

	TF_DeleteSession(sess, status);
	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't delete session" << std::endl;
		TF_DeleteStatus(status);
		return false;
	}

	TF_DeleteStatus(status);

	return true;
}

bool runSession(
	TF_Session* sess,
	const TF_Output* inputOps, TF_Tensor* const* inputTensors, std::size_t numInputs,
	const TF_Output* outputOps, TF_Tensor** outputTensors, std::size_t numOutputs)
{
	if (sess == nullptr || inputOps == nullptr || inputTensors == nullptr || outputOps == nullptr || outputTensors == nullptr)
	{
		return false;
	}

	TF_Status* status = TF_NewStatus();

	TF_SessionRun(
		sess,
		nullptr, // Run options
		inputOps, inputTensors, static_cast<int>(numInputs),
		outputOps, outputTensors, static_cast<int>(numOutputs),
		nullptr, 0, // Target oprations, number of targets
		nullptr, // Run metadata
		status // Output status
	);

	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't run session" << std::endl;
		return false;
	}

	return true;
}

// --------------------------------------------------------
// Tensor
// --------------------------------------------------------
TF_Tensor* createTensor(
	TF_DataType dataType,
	const std::int64_t* dims, std::size_t numDims,
	const void* data, std::size_t len)
{
	if (dims == nullptr || data == nullptr)
	{
		std::cerr << "Error: dims or data is null" << std::endl;
		return nullptr;
	}

	TF_Tensor* tensor = TF_AllocateTensor(dataType, dims, static_cast<int>(numDims), len);
	if (tensor == nullptr)
	{
		std::cerr << "Error: Can't allocate tensor" << std::endl;
		return nullptr;
	}

	void* tensorData = TF_TensorData(tensor);
	if (tensorData == nullptr)
	{
		std::cerr << "Error: Can't create tensor" << std::endl;
		TF_DeleteTensor(tensor);
		return nullptr;
	}

	std::memcpy(tensorData, data, std::min(len, TF_TensorByteSize(tensor)));

	return tensor;
}

void DeleteTensor(TF_Tensor* tensor)
{
	if (tensor == nullptr)
	{
		std::cerr << "Error: Tried to delete tensor, but tensor is null" << std::endl;
		return;
	}
	TF_DeleteTensor(tensor);
}

} // namespace tfutils
