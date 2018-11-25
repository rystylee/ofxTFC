#pragma once

#include <iostream>
#include <string>

#include "ofImage.h"
#include "ofFbo.h"

#include "TFUtils.hpp"
#include "TFInfoUtils.hpp"

class TFModel
{
public:
	TFModel();
	~TFModel();
	
	void setup(
		const std::string& graphPath,
		const std::string& inputOpName,
		const std::string& outputOpName,
		const std::vector<std::int64_t>& inputDims);

	void runImgToImg(const std::vector<ofFloatImage>& inputs, std::vector<ofFloatImage>& outputs);
	//void runVecToImg(const std::vector<std::vector<float>>& inputs, std::vector<ofFloatImage>& outputs);

	// Getter
	TF_Graph* getGraph() { return mGraph; }
	
	// Utils
	void tensorToImg(const std::vector<TF_Tensor*>& tensors, std::vector<ofFloatImage>& imgs);
	void createTensorFromImg(std::vector<TF_Tensor*>& tensors, const std::vector<ofFloatImage>& imgs);
	void printOpInfo() { tfutils::printOp(mGraph); }

private:
	TF_Graph* mGraph;
	TF_Session* mSess;

	std::vector<TF_Output> mInputOps;
	std::vector<TF_Output> mOutputOps;
	std::vector<std::int64_t> mInputDims;
	std::size_t mInputLen { 1 };
};