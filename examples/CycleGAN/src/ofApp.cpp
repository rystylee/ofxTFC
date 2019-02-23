#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);

    mGraphPath.listDir("models");

    //mModel.setup(ofFilePath::getAbsolutePath(mGraphPath.getPath(0)), mInputOpName, mOutputOpName, mInputDims, mModelInputRange, mModelOutputRange);
    string modelPath = "models/horse2zebra.pb";
    //string modelPath = "models/apple2orange.pb";
    mHorse2ZebraModel.setup(ofFilePath::getAbsolutePath(modelPath), mInputOpName, "G_7/output/Tanh", mInputDims, mModelInputRange, mModelOutputRange);
    modelPath = "models/zebra2horse.pb";
    //modelPath = "models/orange2apple.pb";
    mZebra2HorseModel.setup(ofFilePath::getAbsolutePath(modelPath), mInputOpName, mOutputOpName, mInputDims, mModelInputRange, mModelOutputRange);

    ofFloatImage inputImg;
    inputImg.load("images/src.jpg");
    mInputImg.push_back(inputImg);

    ofFloatImage outputImg;
    outputImg.allocate(mInputDims[0], mInputDims[1], OF_IMAGE_COLOR);
    mOutputImg.push_back(outputImg);
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
        mHorse2ZebraModel.runImgToImg(mInputImg, mOutputImg, mImageInputRange, mImageOutputRange);
    else
        mZebra2HorseModel.runImgToImg(mOutputImg, mInputImg, mImageInputRange, mImageOutputRange);

    //mHorse2ZebraModel.runImgToImg(mInputImg, mOutputImg, mImageInputRange, mImageOutputRange);
    //mZebra2HorseModel.runImgToImg(mInputImg, mOutputImg, mImageInputRange, mImageOutputRange);
    
    mOutputImg[0].draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());

    //std::swap(mInputImg, mOutputImg);
}

//--------------------------------------------------------------