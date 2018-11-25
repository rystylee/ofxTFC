#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
	ofSetBackgroundColor(0);

	mGraphPath.listDir("models");

	mModel.setup(ofFilePath::getAbsolutePath(mGraphPath.getPath(0)), mInputOpName, mOutputOpName, mInputDims);

	ofFloatImage  inputImg;
	inputImg.load("images/src.png");
	mInputImg.push_back(inputImg);

	ofFloatImage  outputImg;
	outputImg.allocate(mInputDims[1], mInputDims[2], OF_IMAGE_COLOR);
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
	mModel.runImgToImg(mInputImg, mOutputImg);
	
	mOutputImg[0].draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());

	std::swap(mInputImg, mOutputImg);
}

//--------------------------------------------------------------
