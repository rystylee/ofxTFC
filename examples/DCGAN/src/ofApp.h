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

private:
    void automate(const float timescale);

    ofDirectory mGraphPath;
    TFModel mModel;

    vector<vector<float>> mInputVecs;
    vector<ofFloatImage> mOutputImgs;
    vector<vector<float>> mTargetZ;

    const int mBatchSize { 1 };
    const int mZDimension { 100 };
    const glm::vec2 mGANResolution { 128, 128 };

    const string mInputOpName { "z" };
    const string mOutputOpName { "generator_1/Tanh" };
    const vector<int64_t> mInputDims { static_cast<int64_t>(mBatchSize), mZDimension };

    const glm::vec2 mModelInputRange { -1.0, 1.0 };
    const glm::vec2 mModelOutputRange { -1.0, 1.0 };

    const glm::vec2 mImageInputRange { 0.0, 1.0 };
    const glm::vec2 mImageOutputRange { 0.0, 1.0 };
};