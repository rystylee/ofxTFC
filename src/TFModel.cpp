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
	const std::vector<std::int64_t> inputDims)
{
	// Graph
	mGraph = tfutils::loadGraphDef(graphPath.c_str());
	if (mGraph == nullptr) return;

	// Session
	mSess = tfutils::createSession(mGraph);

	// Operation
	mInputOp = tfutils::loadOperation(mGraph, inputOpName.c_str());
	mOutputOp = tfutils::loadOperation(mGraph, outputOpName.c_str());

	mInputDims = inputDims;
	for (const auto& e : mInputDims)
	{
		mInputLen *= e;
	}
	mInputLen *= sizeof(float);
}

void TFModel::run(ofFloatImage& inputImg, ofFloatImage& outputImg)
{
	TF_Tensor* inputTensor = tfutils::createTensor(
		TF_FLOAT,
		mInputDims.data(), mInputDims.size(),
		inputImg.getPixels().getData(), mInputLen
	);
	TF_Tensor* outputTensor = nullptr;

	const bool success = tfutils::runSession(
		mSess,
		&mInputOp, &inputTensor, 1,
		&mOutputOp, &outputTensor, 1
	);

	if (success) tensorToImg(outputTensor, outputImg);
	else std::cerr << "Error: Can't run session" << std::endl;

	TF_DeleteTensor(inputTensor);
	TF_DeleteTensor(outputTensor);
}

void TFModel::tensorToImg(const TF_Tensor* src, ofFloatImage& dst)
{
	auto data = static_cast<float*>(TF_TensorData(src));
	dst.setFromPixels(data, dst.getWidth(), dst.getHeight(), dst.getImageType(), true);
}
