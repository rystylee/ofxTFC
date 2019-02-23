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
    vector<ofFloatImage> mInputImg;
    vector<ofFloatImage> mOutputImg;

    ofDirectory mGraphPath;

    // ofxTFC
    TFModel mModel;

    const string mInputOpName { "generator/generator_inputs" };
    const string mOutputOpName { "generator/generator_outputs" };
    const vector<int64_t> mInputDims { 1, 256, 256, 3 };
    
    const glm::vec2 mModelInputRange { -1.0, 1.0 };
    const glm::vec2 mModelOutputRange { -1.0, 1.0 };

    const glm::vec2 mImageInputRange { 0.0, 1.0 };
    const glm::vec2 mImageOutputRange { 0.0, 1.0 };
};