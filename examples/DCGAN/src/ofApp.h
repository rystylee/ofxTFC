#pragma once

#include "ofMain.h"
#include "ofxTFC.h"

class ofApp : public ofBaseApp
{
public:
	void setup();
	void update();
	void draw();
	void keyPressed(int key);

	void updateZ();
	void randomizeInput();

	vector<ofFloatImage> mInputImg;
	vector<ofFloatImage> mOutputImg;

	unique_ptr<float[]> mZ;
	unique_ptr<float[]> mTargetZ;

	ofDirectory mGraphPath;

	// ofxTFC
	TFModel mModel;

	const string mInputOpName { "z" };
	const string mOutputOpName { "generator_1/Tanh" };
	const vector<int64_t> mInputDims { 64, 100 };
};
