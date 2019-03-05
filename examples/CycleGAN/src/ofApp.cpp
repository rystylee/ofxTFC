#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);

    mGraphPath.listDir("models");

    string modelPath = "models/horse2zebra.pb";
    mHorse2ZebraModel.init(ofFilePath::getAbsolutePath(modelPath), mInputOpName, "G_7/output/Tanh", mBatchSize, mInputDims, mModelInputRange, mModelOutputRange);
    modelPath = "models/zebra2horse.pb";
    mZebra2HorseModel.init(ofFilePath::getAbsolutePath(modelPath), mInputOpName, mOutputOpName, mBatchSize, mInputDims, mModelInputRange, mModelOutputRange);

    ofFloatImage inputImg;
    inputImg.load("images/src.jpg");
    mInputImgs.push_back(inputImg);

    ofFloatImage outputImg;
    outputImg.allocate(mInputDims[0], mInputDims[1], OF_IMAGE_COLOR);
    mOutputImgs.push_back(outputImg);
}

//--------------------------------------------------------------
void ofApp::update()
{
    ofSetWindowTitle(ofToString(ofGetFrameRate()));
}

//--------------------------------------------------------------
void ofApp::draw()
{
    if (ofGetFrameNum() % 2 == 0)
        mHorse2ZebraModel.runImgsToImgs(mInputImgs, mOutputImgs, mImageInputRange, mImageOutputRange);
    else
        mZebra2HorseModel.runImgsToImgs(mOutputImgs, mInputImgs, mImageInputRange, mImageOutputRange);

    mOutputImgs[0].draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());
}

//--------------------------------------------------------------