#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);

    mGraphPath.listDir("models");
    mModel.init(ofFilePath::getAbsolutePath(mGraphPath.getPath(0)), mInputOpName, mOutputOpName, mBatchSize ,mInputDims, mModelInputRange, mModelOutputRange);

    ofFloatImage inputImg;
    inputImg.allocate(mInputDims[1], mInputDims[2], OF_IMAGE_COLOR);
    mInputImgs.push_back(inputImg);

    ofFloatImage outputImg;
    outputImg.allocate(mInputDims[1], mInputDims[2], OF_IMAGE_COLOR);
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
    mModel.runImgsToImgs(mInputImgs, mOutputImgs, mImageInputRange, mImageOutputRange);
    mOutputImgs[0].draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());
    std::swap(mInputImgs, mOutputImgs);
}

//--------------------------------------------------------------