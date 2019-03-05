#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetBackgroundColor(0);

    mGraphPath.listDir("models");
    mModel.init(ofFilePath::getAbsolutePath(mGraphPath.getPath(0)), mInputOpName, mOutputOpName, mBatchSize, mInputDims, mModelInputRange, mModelOutputRange);

    for (int i = 0; i < mBatchSize; i++)
    {
        ofFloatImage inputImg;
        inputImg.allocate(1, 100, OF_IMAGE_GRAYSCALE);
        mInputImgs.push_back(inputImg);

        ofFloatImage  outputImg;
        outputImg.allocate(mGANResolution.x, mGANResolution.y, OF_IMAGE_COLOR);
        mOutputImgs.push_back(outputImg);

        vector<float> z(mInputDims[1]);
        mZ.emplace_back(z);
        mTargetZ.emplace_back(z);
    }

    //mModel.printOpInfo();
}

//--------------------------------------------------------------
void ofApp::update()
{
    ofSetWindowTitle(ofToString(ofGetFrameRate()));

    //updateZ();
    //if (ofGetFrameNum() % 300 == 0) randomizeInput();

    float t = ofGetElapsedTimef() * 0.1;
    automate(t);
}

//--------------------------------------------------------------
void ofApp::draw()
{
    mModel.runImgsToImgs(mInputImgs, mOutputImgs, mImageInputRange, mImageOutputRange);

    //const int widthNum = 6;
    //const int heightNum = 4;
    //float w = ofGetWidth() / static_cast<float>(widthNum);
    //float h = ofGetHeight() / static_cast<float>(heightNum);

    //for (int x = 0; x < widthNum; x++)
    //{
    //    for (int y = 0; y < heightNum; y++)
    //    {
    //        int index = x * heightNum + y;
    //        mOutputImgs[index].draw(glm::vec2(x * w, y * h), w, h);
    //    }
    //}
    mOutputImgs[0].draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
}

//--------------------------------------------------------------
void ofApp::updateZ()
{
    for (int batch = 0; batch < mBatchSize; batch++)
    {
        for (int i = 0; i < mInputDims[1]; i++)
        {
            mZ[batch][i] += (mTargetZ[batch][i] - mZ[batch][i]) * 0.01;
        }
        mInputImgs[batch].setFromPixels(mZ[batch].data(), 1, mInputDims[1], OF_IMAGE_GRAYSCALE, true);
    }
}

//--------------------------------------------------------------
void ofApp::randomizeInput()
{
    for (int batch = 0; batch < mBatchSize; batch++)
    {
        for (int i = 0; i < mInputDims[1]; i++)
        {
            mTargetZ[batch][i] = ofRandom(-1.0, 1.0);
        }
    }
}

//--------------------------------------------------------------
void ofApp::automate(const float timescale)
{
    for (int batch = 0; batch < mBatchSize; batch++)
    {
        for (int i = 0; i < mInputDims[1]; i++)
        {
            int index = batch * mInputDims[1] + i;
            //float n = ofMap(ofNoise(static_cast<float>(index) * mZScale, mTime * mTimeScale), 0.0, 1.0, -1.0, 1.0);
            float n = ofMap(ofNoise(static_cast<float>(index) * 0.1, timescale), 0.0, 1.0, -1.0, 1.0);
            mZ[batch][i] = n;
        }
        mInputImgs[batch].setFromPixels(mZ[batch].data(), 1, mInputDims[1], OF_IMAGE_GRAYSCALE, true);
    }
}

//--------------------------------------------------------------