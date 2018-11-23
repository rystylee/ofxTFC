#pragma once

#include <iostream>
#include <string>

#include "ofImage.h"

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
		const std::vector<std::int64_t> inputDims);

	void run(ofFloatImage& inputImg, ofFloatImage& outputImg);

	void tensorToImg(const TF_Tensor* src, ofFloatImage& dst);

	// Debug
	void printOpInfo() { tfutils::printOp(mGraph); }

private:
	TF_Graph* mGraph;
	TF_Session* mSess;

	TF_Output mInputOp;
	TF_Output mOutputOp;

	std::vector<std::int64_t> mInputDims;
	std::size_t mInputLen { 1 };
};