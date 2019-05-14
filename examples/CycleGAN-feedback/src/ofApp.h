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
    ofDirectory mGraphPath;
    TFModel mX2YModel;
    TFModel mY2XModel;

    vector<ofFloatImage> mInputImgs;
    vector<ofFloatImage> mOutputImgs;

    const string mInputOpName { "input_image" };
    const string mX2YOutputOpName { "G_7/output/Tanh" };
    const string mY2XOutputOpName { "F_7/output/Tanh" };
    const int mBatchSize { 1 };
    const vector<int64_t> mInputDims { 256, 256, 3 };
    
    const glm::vec2 mModelInputRange { -1.0, 1.0 };
    const glm::vec2 mModelOutputRange { -1.0, 1.0 };

    const glm::vec2 mImageInputRange { 0.0, 1.0 };
    const glm::vec2 mImageOutputRange { 0.0, 1.0 };
};