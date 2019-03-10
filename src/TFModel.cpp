#include "TFModel.hpp"

// --------------------------------------------------------
// Public
// --------------------------------------------------------
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

void TFModel::init(
    const std::string& graphPath,
    const std::string& inputOpName,
    const std::string& outputOpName,
    const int batchSize,
    const std::vector<std::int64_t>& inputDims,
    const glm::vec2& modelInputRange,
    const glm::vec2& modelOutputRange,
    const tfutils::SessionConfigType& sessionConfigType
)
{
    // Graph
    mGraph = tfutils::loadGraphDef(graphPath.c_str());
    if (mGraph == nullptr) return;

    // Session
    mSess = tfutils::createSession(mGraph, sessionConfigType);

    // Batch size
    mBatchSize = batchSize;

    // Input dimension
    mInputDims = inputDims;

    // Operation
    mInputOps = tfutils::loadOperations(mGraph, inputOpName.c_str());
    mOutputOps = tfutils::loadOperations(mGraph, outputOpName.c_str());

    // Model input/output range
    mModelInputRange = modelInputRange;
    mModelOutputRange = modelOutputRange;
    
    // Input Length
    for (auto& e : mInputDims)
    {
        mInputLen *= e;
    }
    mInputLen *= sizeof(float);

    // Input BytesNum
    mInputBytesNum = (mInputLen / static_cast<size_t>(mBatchSize) / sizeof(float));
}

// --------------------------------------------------------
// Private
// --------------------------------------------------------
void TFModel::runImgsToImgs(const std::vector<ofFloatImage>& inputs, std::vector<ofFloatImage>& outputs, const glm::vec2& imageInputRange, const glm::vec2& imageOutputRange)
{
    std::vector<TF_Tensor*> inputTensors;
    imgsToTensors(inputTensors, inputs, imageInputRange);
    std::vector<TF_Tensor*> outputTensors = { nullptr };

    const bool success = tfutils::runSession(
        mSess,
        mInputOps, inputTensors,
        mOutputOps, outputTensors
    );

    if (success) tensorsToImgs(outputTensors, outputs, imageOutputRange);

    tfutils::deleteTensors(inputTensors);
    tfutils::deleteTensors(outputTensors);
}

void TFModel::runFbosToImgs(const std::vector<ofFbo>& inputs, std::vector<ofFloatImage>& outputs, const glm::vec2& imageInputRange, const glm::vec2& imageOutputRange)
{
    std::vector<TF_Tensor*> inputTensors;
    fbosToTensors(inputTensors, inputs, imageInputRange);
    std::vector<TF_Tensor*> outputTensors = { nullptr };

    const bool success = tfutils::runSession(
        mSess,
        mInputOps, inputTensors,
        mOutputOps, outputTensors
    );

    if (success) tensorsToImgs(outputTensors, outputs, imageOutputRange);

    tfutils::deleteTensors(inputTensors);
    tfutils::deleteTensors(outputTensors);
}

void TFModel::runFbosToFbos(const std::vector<ofFbo>& inputs, std::vector<ofFbo>& outputs, const glm::vec2& fboInputRange, const glm::vec2& fboOutputRange)
{
    std::vector<TF_Tensor*> inputTensors;
    fbosToTensors(inputTensors, inputs, fboInputRange);
    std::vector<TF_Tensor*> outputTensors = { nullptr };

    const bool success = tfutils::runSession(
        mSess,
        mInputOps, inputTensors,
        mOutputOps, outputTensors
    );

    if (success) tensorsToFbos(outputTensors, outputs, fboOutputRange);

    tfutils::deleteTensors(inputTensors);
    tfutils::deleteTensors(outputTensors);
}

void TFModel::runVecsToImgs(const std::vector<std::vector<float>>& inputs, std::vector<ofFloatImage>& outputs, const glm::vec2& imageInputRange, const glm::vec2& imageOutputRange)
{
    std::vector<TF_Tensor*> inputTensors;
    vecsToTensors(inputTensors, inputs, imageInputRange);
    std::vector<TF_Tensor*> outputTensors = { nullptr };

    const bool success = tfutils::runSession(
        mSess,
        mInputOps, inputTensors,
        mOutputOps, outputTensors
    );

    if (success) tensorsToImgs(outputTensors, outputs, imageOutputRange);

    tfutils::deleteTensors(inputTensors);
    tfutils::deleteTensors(outputTensors);
}

void TFModel::imgsToTensors(std::vector<TF_Tensor*>& tensors, const std::vector<ofFloatImage>& imgs, const glm::vec2& imageInputRange)
{
    // At this time, tensors.size() is always 1
    std::vector<float> inputBuffer;
    inputBuffer.resize(mInputLen / sizeof(float));
    size_t offset = 0;
    for (const auto& img : imgs)
    {
        std::vector<float> buffer;
        buffer.resize(mInputBytesNum);
        std::memcpy(buffer.data(), img.getPixels().getData(), mInputBytesNum * sizeof(float));
        tfutils::map(buffer, imageInputRange.x, imageInputRange.y, mModelInputRange.x, mModelInputRange.y);
        inputBuffer.insert(inputBuffer.begin() + offset, buffer.begin(), buffer.end());
        offset += mInputBytesNum;
    }
    tensors.emplace_back( tfutils::createTensor(TF_FLOAT, mInputDims.data(), mInputDims.size(), inputBuffer.data(), mInputLen) );
}

void TFModel::fbosToTensors(std::vector<TF_Tensor*>& tensors, const std::vector<ofFbo>& fbos, const glm::vec2& fboInputRange)
{
    // At this time, tensors.size() is always 1
    std::vector<float> inputBuffer;
    inputBuffer.resize(mInputLen / sizeof(float));
    size_t offset = 0;
    for (const auto& fbo : fbos)
    {
        ofFloatImage img;
        fbo.readToPixels(img.getPixels());
        std::vector<float> buffer;
        buffer.resize(mInputBytesNum);
        std::memcpy(buffer.data(), img.getPixels().getData(), mInputBytesNum * sizeof(float));
        tfutils::map(buffer, fboInputRange.x, fboInputRange.y, mModelInputRange.x, mModelInputRange.y);
        inputBuffer.insert(inputBuffer.begin() + offset, buffer.begin(), buffer.end());
        offset += mInputBytesNum;
    }
    tensors.emplace_back( tfutils::createTensor(TF_FLOAT, mInputDims.data(), mInputDims.size(), inputBuffer.data(), mInputLen) );
}

void TFModel::vecsToTensors(std::vector<TF_Tensor*>& tensors, const std::vector<std::vector<float>>& vecs, const glm::vec2& vecInputRange)
{
    // At this time, tensors.size() is always 1
    std::vector<float> inputBuffer;
    inputBuffer.resize(mInputLen / sizeof(float));
    size_t offset = 0;
    for (const auto& vec : vecs)
    {
        std::vector<float> buffer;
        buffer.resize(mInputBytesNum);
        std::memcpy(buffer.data(), vec.data(), mInputBytesNum * sizeof(float));
        tfutils::map(buffer, vecInputRange.x, vecInputRange.y, mModelInputRange.x, mModelInputRange.y);
        inputBuffer.insert(inputBuffer.begin() + offset, buffer.begin(), buffer.end());
        offset += mInputBytesNum;
    }
    tensors.emplace_back( tfutils::createTensor(TF_FLOAT, mInputDims.data(), mInputDims.size(), inputBuffer.data(), mInputLen) );
}

void TFModel::tensorsToImgs(const std::vector<TF_Tensor*>& tensors, std::vector<ofFloatImage>& imgs, const glm::vec2& imageOutputRange)
{
    // At this time, data.size() is always 1
    std::vector<std::vector<float>> data = tfutils::tensorData<float>(tensors);

    tfutils::map(data, mModelOutputRange.x, mModelOutputRange.y, imageOutputRange.x, imageOutputRange.y);

    const size_t bytesNum = static_cast<size_t>(imgs[0].getWidth() * imgs[0].getHeight() * 3);
    size_t offset = 0;
    for (auto& img : imgs)
    {
        std::vector<float> d;
        d.resize(bytesNum);
        d.insert(d.begin(), data[0].begin() + offset, data[0].begin() + offset + bytesNum);
        img.setFromPixels(d.data(), img.getWidth(), img.getHeight(), img.getImageType(), true);
        offset += bytesNum;
    }
}

void TFModel::tensorsToFbos(const std::vector<TF_Tensor*>& tensors, std::vector<ofFbo>& fbos, const glm::vec2& fboOutputRange)
{
    // At this time, data.size() is always 1
    std::vector<std::vector<float>> data = tfutils::tensorData<float>(tensors);

    tfutils::map(data, mModelOutputRange.x, mModelOutputRange.y, fboOutputRange.x, fboOutputRange.y);

    const size_t bytesNum = static_cast<size_t>(fbos[0].getWidth() * fbos[0].getHeight() * 3);
    size_t offset = 0;
    for (auto& fbo : fbos)
    {
        std::vector<float> d;
        d.resize(bytesNum);
        d.insert(d.begin(), data[0].begin() + offset, data[0].begin() + offset + bytesNum);
        ofFloatPixels pix;
        pix.setFromPixels(d.data(), fbo.getWidth(), fbo.getHeight(), OF_IMAGE_COLOR);
        fbo.getTexture().loadData(pix, GL_RGB);
        offset += bytesNum;
    }
}