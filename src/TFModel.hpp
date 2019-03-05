#pragma once

#include <iostream>
#include <string>

#include "ofImage.h"
#include "ofFbo.h"
#include "glm/glm.hpp"

#include "TFUtils.hpp"
#include "TFInfoUtils.hpp"
#include "MathUtils.hpp"

class TFModel
{
public:
    TFModel();
    ~TFModel();
    
    void init(
        const std::string& graphPath,
        const std::string& inputOpName,
        const std::string& outputOpName,
        const int batchSize,
        const std::vector<std::int64_t>& inputDims,
        const glm::vec2& modelInputRange,
        const glm::vec2& modelOutputRange,
        const tfutils::SessionConfigType& sessionConfigType = tfutils::SessionConfigType::NONE);

    void runImgsToImgs(const std::vector<ofFloatImage>& inputs, std::vector<ofFloatImage>& outputs, const glm::vec2& imageInputRange, const glm::vec2& imageOutputRange);
    void runFbosToImgs(const std::vector<ofFbo>& inputs, std::vector<ofFloatImage>& outputs, const glm::vec2& imageInputRange, const glm::vec2& imageOutputRange);
    void runFbosToFbos(const std::vector<ofFbo>& inputs, std::vector<ofFbo>& outputs, const glm::vec2& fboInputRange, const glm::vec2& fboOutputRange);

    // Getter
    TF_Graph* getGraph() { return mGraph; }
    
    // Utils
    void imgsToTensors(std::vector<TF_Tensor*>& tensors, const std::vector<ofFloatImage>& imgs, const glm::vec2& imageInputRange);
    void fbosToTensors(std::vector<TF_Tensor*>& tensors, const std::vector<ofFbo>& fbos, const glm::vec2& fboInputRange);
    void tensorsToImgs(const std::vector<TF_Tensor*>& tensors, std::vector<ofFloatImage>& imgs, const glm::vec2& imageOutputRange);
    void tensorsToFbos(const std::vector<TF_Tensor*>& tensors, std::vector<ofFbo>& fbos, const glm::vec2& fboOutputRange);
    void printOpInfo() { tfutils::printOp(mGraph); }

private:
    TF_Graph* mGraph;
    TF_Session* mSess;

    std::vector<TF_Output> mInputOps;
    std::vector<TF_Output> mOutputOps;

    int mBatchSize { 1 };
    std::vector<std::int64_t> mInputDims;
    size_t mInputBytesNum { 1 };
    size_t mOutputBytesNum { 1 };

    glm::vec2 mModelInputRange;
    glm::vec2 mModelOutputRange;
    std::size_t mInputLen { 1 };
};