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
    TFModel mModel;

    const int mBatchSize { 1 };
    const glm::vec2 mGANResolution { 128, 128 };

    const string mInputOpName { "z" };
    const string mOutputOpName { "generator_1/Tanh" };
    const vector<int64_t> mInputDims { mBatchSize, 100 };

    const glm::vec2 mModelInputRange { -1.0, 1.0 };
    const glm::vec2 mModelOutputRange { -1.0, 1.0 };

    const glm::vec2 mImageInputRange { 0.0, 1.0 };
    const glm::vec2 mImageOutputRange { 0.0, 1.0 };
};