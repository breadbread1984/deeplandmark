#ifndef LANDMARKER_H
#define LANDMARKER_H

#include <iostream>
#include <string>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <opencv2/opencv.hpp>
#include "Regressor.h"

using namespace std;
using namespace cv;

class Landmarker {
	static const string modelnames[10];
	vector<Regressor> level1;
	vector<Regressor> level2;
	vector<Regressor> level3;
protected:
	vector<float> level(Mat facegrayimg, vector<int> bounding, vector<float> landmark, vector<Regressor> cnns, pair<float,float> padding);
	boost::tuple<Mat,vector<int> > getPatch(Mat img, vector<int> bounding, Point2f pts, float padding,Size sz);
	Mat crop(Mat img, vector<int> bounding, Size size);
	vector<float> project(vector<int> bounding,vector<float> pos);
	vector<float> reproject(vector<int> bounding,vector<float> pos);
public:
	Landmarker();
	~Landmarker();
	vector<Point> detectLandmark(Mat img,Rect face);
};

#endif
