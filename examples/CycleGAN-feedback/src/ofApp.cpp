#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);
    ofSetVerticalSync(false);

    mGraphPath.listDir("models");

    string modelPath = "models/horse2zebra.pb";
    mX2YModel.init(ofFilePath::getAbsolutePath(modelPath), mInputOpName, mX2YOutputOpName, mBatchSize, mInputDims, mModelInputRange, mModelOutputRange);
    modelPath = "models/zebra2horse.pb";
    mY2XModel.init(ofFilePath::getAbsolutePath(modelPath), mInputOpName, mY2XOutputOpName, mBatchSize, mInputDims, mModelInputRange, mModelOutputRange);

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
        mX2YModel.runImgsToImgs(mInputImgs, mOutputImgs, mImageInputRange, mImageOutputRange);
    else
        mY2XModel.runImgsToImgs(mOutputImgs, mInputImgs, mImageInputRange, mImageOutputRange);

    mOutputImgs[0].draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());
}

//--------------------------------------------------------------