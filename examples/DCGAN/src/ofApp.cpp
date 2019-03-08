#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);
    ofSetVerticalSync(false);

    mGraphPath.listDir("models");
    mModel.init(ofFilePath::getAbsolutePath(mGraphPath.getPath(0)), mInputOpName, mOutputOpName, mBatchSize, mInputDims, mModelInputRange, mModelOutputRange);

    for (int i = 0; i < mBatchSize; i++)
    {
        vector<float> z(mZDimension);
        mInputVecs.emplace_back(z);
        mTargetZ.emplace_back(z);

        ofFloatImage  outputImg;
        outputImg.allocate(mGANResolution.x, mGANResolution.y, OF_IMAGE_COLOR);
        mOutputImgs.push_back(outputImg);
    }

    //mModel.printOpInfo();
}

//--------------------------------------------------------------
void ofApp::update()
{
    ofSetWindowTitle(ofToString(ofGetFrameRate()));

    float t = ofGetElapsedTimef() * 0.1;
    automate(t);
}

//--------------------------------------------------------------
void ofApp::draw()
{
    mModel.runVecsToImgs(mInputVecs, mOutputImgs, mImageInputRange, mImageOutputRange);
    mOutputImgs[0].draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
}

//--------------------------------------------------------------
void ofApp::automate(const float timescale)
{
    for (int batch = 0; batch < mBatchSize; batch++)
    {
        for (int i = 0; i < mZDimension; i++)
        {
            int index = batch * mInputDims[1] + i;
            float n = ofMap(ofNoise(static_cast<float>(index) * 0.1, timescale), 0.0, 1.0, -1.0, 1.0);
            mInputVecs[batch][i] = n;
        }
    }
}

//--------------------------------------------------------------