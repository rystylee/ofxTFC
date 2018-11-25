#include "TFModel.hpp"

TFModel::TFModel()
{
	std::cout << "Tensorflow Version: " << TF_Version() << std::endl;
}

TFModel::~TFModel()
{
	TF_Status* status = TF_NewStatus();

	// Session
	TF_CloseSession(mSess, status);
	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't close session" << std::endl;
		TF_CloseSession(mSess, status);
		TF_DeleteSession(mSess, status);
		TF_DeleteStatus(status);
	}

	TF_DeleteSession(mSess, status);
	if (TF_GetCode(status) != TF_OK)
	{
		std::cerr << "Error: Can't delete session" << std::endl;
		TF_DeleteStatus(status);
	}

	// Graph
	TF_DeleteGraph(mGraph);
	// Status
	TF_DeleteStatus(status);
}

void TFModel::setup(
	const std::string& graphPath,
	const std::string& inputOpName,
	const std::string& outputOpName,
	const std::vector<std::int64_t>& inputDims)
{
	// Graph
	mGraph = tfutils::loadGraphDef(graphPath.c_str());
	if (mGraph == nullptr) return;

	// Session
	mSess = tfutils::createSession(mGraph);

	// Operation
	mInputOps = tfutils::loadOperations(mGraph, inputOpName.c_str());
	mOutputOps = tfutils::loadOperations(mGraph, outputOpName.c_str());

	// Input dimension
	mInputDims = inputDims;
	
	// Input Length
	for (auto& e : mInputDims)
	{
		mInputLen *= e;
	}
	mInputLen *= sizeof(float);
}

// --------------------------------------------------------
// Utils
// --------------------------------------------------------
void TFModel::runImgToImg(const std::vector<ofFloatImage>& inputs, std::vector<ofFloatImage>& outputs)
{
	std::vector<TF_Tensor*> inputTensors;
	createTensorFromImg(inputTensors, inputs);
	std::vector<TF_Tensor*> outputTensors = { nullptr };

	const bool success = tfutils::runSession(
		mSess,
		mInputOps, inputTensors,
		mOutputOps, outputTensors
	);

	if (success) tensorToImg(outputTensors, outputs);

	tfutils::deleteTensors(inputTensors);
	tfutils::deleteTensors(outputTensors);
}

void TFModel::tensorToImg(const std::vector<TF_Tensor*>& tensors, std::vector<ofFloatImage>& imgs)
{
	const std::vector<std::vector<float>> data = tfutils::tensorData<float>(tensors);

	if (imgs.size() != 1)
	{
		std::cerr << "Warning: You have to specify the index of output tensors" << std::endl;
	}

	imgs[0].setFromPixels(data[0].data(), imgs[0].getWidth(), imgs[0].getHeight(), imgs[0].getImageType(), true);
}

void TFModel::createTensorFromImg(std::vector<TF_Tensor*>& tensors, const std::vector<ofFloatImage>& imgs)
{
	std::vector<float> inputBuffer;
	inputBuffer.resize(mInputLen / sizeof(float));
	std::memcpy(inputBuffer.data(), imgs[0].getPixels().getData(), mInputLen);

	tensors = { tfutils::createTensor(TF_FLOAT, mInputDims.data(), mInputDims.size(), inputBuffer.data(), mInputLen) };
}
