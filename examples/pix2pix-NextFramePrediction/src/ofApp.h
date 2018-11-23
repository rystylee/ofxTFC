#pragma once

#include "ofMain.h"
#include "ofxTFC.h"

class ofApp : public ofBaseApp
{
public:
	void setup();	
	void update();
	void draw();
		
private:
	ofFloatImage mInputImg;
	ofFloatImage mOutputImg;

	ofDirectory mGraphPath;

	// ofxTFC
	TFModel mModel;

	const string mInputOpName { "generator/generator_inputs" };
	const string mOutputOpName { "generator/generator_outputs" };
	const vector<int64_t> mInputDims { 1, 256, 256, 3 };
};
