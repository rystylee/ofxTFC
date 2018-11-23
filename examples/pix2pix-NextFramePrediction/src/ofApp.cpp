#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
	ofSetBackgroundColor(0);

	mGraphPath.listDir("models");

	mModel.setup(ofFilePath::getAbsolutePath(mGraphPath.getPath(0)), mInputOpName, mOutputOpName, mInputDims);

	mInputImg.load("images/src.png");
	mOutputImg.allocate(mInputDims[1], mInputDims[2], OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::update()
{
	ofSetWindowTitle(ofToString(ofGetFrameRate()));
}

//--------------------------------------------------------------
void ofApp::draw()
{
	mModel.run(mInputImg, mOutputImg);
	
	mOutputImg.draw(glm::vec2(0, 0), ofGetWidth(), ofGetHeight());

	std::swap(mInputImg, mOutputImg);
}

//--------------------------------------------------------------
