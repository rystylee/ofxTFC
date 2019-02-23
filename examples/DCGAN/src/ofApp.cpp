#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);

    mGraphPath.listDir("models");

    mModel.setup(ofFilePath::getAbsolutePath(mGraphPath.getPath(0)), mInputOpName, mOutputOpName, mInputDims, mModelInputRange, mModelOutputRange);

    ofFloatImage inputImg;
    inputImg.allocate(mBatchSize, 100, OF_IMAGE_GRAYSCALE);
    mInputImg.push_back(inputImg);

    ofFloatImage  outputImg;
    outputImg.allocate(mGANResolution.x, mGANResolution.y, OF_IMAGE_COLOR);
    mOutputImg.push_back(outputImg);

    mZ= make_unique<float[]>(mInputDims[0] * mInputDims[1]);
    mTargetZ= make_unique<float[]>(mInputDims[0] * mInputDims[1]);
}

//--------------------------------------------------------------
void ofApp::update()
{
    ofSetWindowTitle(ofToString(ofGetFrameRate()));

    updateZ();
    if (ofGetFrameNum() % 300 == 0) randomizeInput();
}

//--------------------------------------------------------------
void ofApp::draw()
{
    mModel.runImgToImg(mInputImg, mOutputImg, mImageInputRange, mImageOutputRange);
    mOutputImg[0].draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
}

//--------------------------------------------------------------
void ofApp::updateZ()
{
    for (int batch = 0; batch < mInputDims[0]; batch++)
    {
        for (int z = 0; z < mInputDims[1]; z++)
        {
            int index = batch * mInputDims[1] + z;
            mZ[index] += (mTargetZ[index] - mZ[index]) * 0.01;
        }
    }
    mInputImg[0].setFromPixels(mZ.get(), mInputDims[0], mInputDims[1], OF_IMAGE_GRAYSCALE, true);
}

//--------------------------------------------------------------
void ofApp::randomizeInput()
{
    const auto v = make_unique<float[]>(mInputDims[0] * mInputDims[1]);
    for (int batch = 0; batch < mInputDims[0]; batch++)
    {
        for (int z = 0; z < mInputDims[1]; z++)
        {
            int index = batch * mInputDims[1] + z;
            mTargetZ[index] = ofRandom(-1.0, 1.0);
        }
    }
}
