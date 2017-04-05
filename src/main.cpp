#include <cstdlib>
#include <iostream>
#include <boost/tuple/tuple.hpp>
#include <opencv2/opencv.hpp>
#include "Landmarker.h"

using namespace std;

int main(int argc, char ** argv)
{
	Landmarker lm;
	VideoCapture vc;
	VideoWriter writer;
	CascadeClassifier detector("model_values/haarcascade_frontalface_default.xml");
	int codec = CV_FOURCC('M','J','P','G');
	double fps = 25.0;
	string filename = "output.avi";
	writer.open(filename,codec,fps,Size(640,480),true);
	vc.open(0);
	if(false == vc.isOpened()) {
		cout<<"摄像机打开错误！"<<endl;
		return EXIT_FAILURE;
	}
	Mat img;
	namedWindow("keypoints",0);
	while(vc.read(img)) {
		vector<Rect> faces;
		detector.detectMultiScale(img,faces);
		if(faces.size()) {
			vector<Point> keypoints = lm.detectLandmark(img,faces[0]);
			rectangle(img,faces[0],Scalar(0,0,255));
			for(vector<Point>::iterator it = keypoints.begin() ; it != keypoints.end() ; it++)
				circle(img,*it,10,Scalar(0,0,255),-1);
			writer.write(img);
			imshow("keypoints",img);
		}
		char k = waitKey(5);
		if(k == 'q') break;
	}
	
	return EXIT_SUCCESS;
}
