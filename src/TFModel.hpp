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
    
    void setup(
        const std::string& graphPath,
        const std::string& inputOpName,
        const std::string& outputOpName,
        const std::vector<std::int64_t>& inputDims,
        const glm::vec2& modelInputRange,
        const glm::vec2& modelOutputRange);

    void runImgToImg(const std::vector<ofFloatImage>& inputs, std::vector<ofFloatImage>& outputs, const glm::vec2& imageInputRange, const glm::vec2& imageOutputRange);
    //void runVecToImg(const std::vector<std::vector<float>>& inputs, std::vector<ofFloatImage>& outputs);

    // Getter
    TF_Graph* getGraph() { return mGraph; }
    
    // Utils
    void imgToTensor(std::vector<TF_Tensor*>& tensors, const std::vector<ofFloatImage>& imgs, const glm::vec2& imageInputRange);
    void tensorToImg(const std::vector<TF_Tensor*>& tensors, std::vector<ofFloatImage>& imgs, const glm::vec2& imageOutputRange);
    void printOpInfo() { tfutils::printOp(mGraph); }

private:
    TF_Graph* mGraph;
    TF_Session* mSess;

    std::vector<TF_Output> mInputOps;
    std::vector<TF_Output> mOutputOps;
    std::vector<std::int64_t> mInputDims;
    glm::vec2 mModelInputRange;
    glm::vec2 mModelOutputRange;
    std::size_t mInputLen { 1 };
};