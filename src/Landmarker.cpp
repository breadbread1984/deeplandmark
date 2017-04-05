#include <stdexcept>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "Landmarker.h"
#include "matrix_basic.hpp"

using namespace boost;
using namespace boost::filesystem;
namespace ublas = boost::numeric::ublas;

const string Landmarker::modelnames[10] = {"LE1", "RE1", "N1", "LM1", "RM1", "LE2", "RE2", "N2", "LM2", "RM2"};

Landmarker::Landmarker()
{
	path deploymodelroot("deploy_model");
	path modelvaluesroot("model_values");
	path trainroot("data/train");
	//加载第一层参数和模型
	level1.push_back(Regressor(
		(deploymodelroot / "1_F.prototxt").string(),
		(modelvaluesroot / "1_F.caffemodel").string()
	));
	level1.push_back(Regressor(
		(deploymodelroot / "1_EN.prototxt").string(),
		(modelvaluesroot / "1_EN.caffemodel").string()
	));
	level1.push_back(Regressor(
		(deploymodelroot / "1_NM.prototxt").string(),
		(modelvaluesroot / "1_NM.caffemodel").string()
	));
	//加载第二三层参数和模型
	for(int i = 0 ; i < 10 ; i++) {
		string modelname2 = string("2_") + modelnames[i];
		string modelname3 = string("3_") + modelnames[i];
		level2.push_back(Regressor(
			(deploymodelroot / (modelname2 + ".prototxt")).string(),
			(modelvaluesroot / (modelname2 + ".caffemodel")).string()
		));
		level3.push_back(Regressor(
			(deploymodelroot / (modelname3 + ".prototxt")).string(),
			(modelvaluesroot / (modelname3 + ".caffemodel")).string()
		));
	}
}

Landmarker::~Landmarker()
{	
}

vector<Point> Landmarker::detectLandmark(Mat img,Rect facearea)
{
	//用[left right top bottom]来表示目标框
	vector<int> facearea_v;
	facearea_v.push_back(facearea.x);
	facearea_v.push_back(facearea.x + facearea.width - 1);
	facearea_v.push_back(facearea.y);
	facearea_v.push_back(facearea.y + facearea.height - 1);
	vector<int> eyesnosearea_v = facearea_v;
	eyesnosearea_v[3] = eyesnosearea_v[2] + (31.0 / 39) * (eyesnosearea_v[3] - eyesnosearea_v[2]);
	vector<int> nosemoutharea_v = facearea_v;
	nosemoutharea_v[2] = nosemoutharea_v[2] + (8.0 / 39) * (nosemoutharea_v[3] - nosemoutharea_v[2]);
	
	Mat gray;
	cvtColor(img,gray,CV_BGR2GRAY);
//	Mat facegrayimg = gray(facearea);
//	resize(facegrayimg,facegrayimg,Size(39,39));
	Mat facegrayimg = crop(gray,facearea_v,Size(39,39));
	//level-1计算出关键点的大概位置
	vector<float> face_landmark = level1[0].Regress(facegrayimg); //五点
	vector<float> eyesnose_landmark = level1[1].Regress(facegrayimg(Rect(0,0,39,31)));//三点
	vector<float> nosemouth_landmark = level1[2].Regress(facegrayimg(Rect(0,8,39,31)));//三点
	assert(face_landmark.size() == 10);
	assert(eyesnose_landmark.size() == 6);
	assert(nosemouth_landmark.size() == 6);
/*
#ifndef NDEBUG
	cout<<"face_landmark = "<<endl;
	for(vector<float>::iterator it = face_landmark.begin() ; it != face_landmark.end() ; it++)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"eyesnose_landmark = "<<endl;
	for(vector<float>::iterator it = eyesnose_landmark.begin() ; it != eyesnose_landmark.end() ; it++)
		cout<<*it<<" ";
	cout<<endl;
	cout<<"nosemouth_landmark = "<<endl;
	for(vector<float>::iterator it = nosemouth_landmark.begin() ; it != nosemouth_landmark.end() ; it++)
		cout<<*it<<" ";
	cout<<endl;

	namedWindow("face_debug",0);
	namedWindow("eyesnose_debug",0);
	namedWindow("nosemouth_debug",0);
	Mat face_debug; facegrayimg.copyTo(face_debug);
	Mat eyesnose_debug; facegrayimg(Rect(0,0,39,31)).copyTo(eyesnose_debug);
	Mat nosemouth_debug; facegrayimg(Rect(0,8,39,31)).copyTo(nosemouth_debug);
	for(int i = 0 ; i < 5 ; i++) {
		vector<float> pos(2);
		pos[0] = face_landmark[2 * i];
		pos[1] = face_landmark[2 * i + 1];
		circle(face_debug,Point(pos[0] * 39,pos[1] * 39),1,Scalar(0,0,255));
	}
	for(int i = 0 ; i < 3 ; i++) {
		vector<float> pos(2);
		pos[0] = eyesnose_landmark[2 * i];
		pos[1] = eyesnose_landmark[2 * i + 1];
		circle(eyesnose_debug,Point(pos[0] * 39,pos[1] * 31),1,Scalar(0,0,255));
	}
	for(int i = 0 ; i < 3 ; i++) {
		vector<float> pos(2);
		pos[0] = nosemouth_landmark[2 * i];
		pos[1] = nosemouth_landmark[2 * i + 1];
		circle(nosemouth_debug,Point(pos[0] * 39,pos[1] * 31),1,Scalar(0,0,255));
	}
	imshow("face_debug",face_debug);
	imshow("eyesnose_debug",eyesnose_debug);
	imshow("nosemouth_debug",nosemouth_debug);
	waitKey();
#endif/*/
#if 1
	//相对坐标变成绝对坐标
	face_landmark = reproject(facearea_v,face_landmark);
	eyesnose_landmark = reproject(eyesnosearea_v,eyesnose_landmark);
	nosemouth_landmark = reproject(nosemoutharea_v,nosemouth_landmark);
	//左眼睛绝对坐标
	face_landmark[0] = (face_landmark[0] + eyesnose_landmark[0]) / 2;
	face_landmark[1] = (face_landmark[1] + eyesnose_landmark[1]) / 2;
	//右眼睛绝对坐标
	face_landmark[2] = (face_landmark[2] + eyesnose_landmark[2]) / 2;
	face_landmark[3] = (face_landmark[3] + eyesnose_landmark[3]) / 2;
	//鼻子绝对坐标
	face_landmark[4] = (face_landmark[4] + eyesnose_landmark[4] + nosemouth_landmark[0]) / 3;
	face_landmark[5] = (face_landmark[5] + eyesnose_landmark[5] + nosemouth_landmark[1]) / 3;
	//左嘴角绝对坐标
	face_landmark[6] = (face_landmark[6] + nosemouth_landmark[2]) / 2;
	face_landmark[7] = (face_landmark[7] + nosemouth_landmark[3]) / 2;
	//右嘴角绝对坐标
	face_landmark[8] = (face_landmark[8] + nosemouth_landmark[4]) / 2;
	face_landmark[9] = (face_landmark[9] + nosemouth_landmark[5]) / 2;
	//坐标变成相对坐标
	face_landmark = project(facearea_v,face_landmark);
#endif

	//level-2更新关键点位置
	face_landmark = level(gray,facearea_v,face_landmark,level2,make_pair(0.16,0.18));
	//level-3更新关键点位置
	face_landmark = level(gray,facearea_v,face_landmark,level3,make_pair(0.11,0.12));
	//转换成绝对坐标
	face_landmark = reproject(facearea_v,face_landmark);
	vector<Point> retVal(5);
	for(int i = 0 ; i < 5 ; i++) {
		vector<float> pos(2);
		pos[0] = face_landmark[2 * i];
		pos[1] = face_landmark[2 * i + 1];
		retVal[i] = Point(pos[0],pos[1]);
	}
	return retVal;
}

vector<float> Landmarker::level(Mat gray, vector<int> bounding, vector<float> landmark, vector<Regressor> cnns, pair<float,float> padding)
{
	//landmark是相对坐标
	assert(gray.type() == CV_8UC1);
	assert(landmark.size() == 10);
	for(int i = 0 ; i < 5 ; i++) {
		//提取第i点的相对坐标
		float x = landmark[2 * i];
		float y = landmark[2 * i + 1];
		//在关键点位置提取第一个尺寸的patch
		boost::tuple<Mat,vector<int> > patch_data = getPatch(gray,bounding,Point2f(x,y),padding.first,Size(15,15));
		Mat patch = get<0>(patch_data);
		vector<int> area_v = get<1>(patch_data);
		vector<float> d1 = cnns[i].Regress(patch);
		assert(2 == d1.size());
		//patch中心相对patch坐标变成相对目标框坐标
		d1 = project(bounding,reproject(area_v,d1));
		//在关键点位置提取第二个尺寸的patch
		patch_data = getPatch(gray,bounding,Point2f(x,y),padding.second,Size(15,15));
		patch = get<0>(patch_data);
		area_v = get<1>(patch_data);
		vector<float> d2 = cnns[i + 5].Regress(patch);
		assert(2 == d2.size());
		//patch中心相对patch坐标变成相对目标框坐标
		d2 = project(bounding,reproject(area_v,d2));
		//两个偏置求平均
		landmark[2 * i] = (d1[0] + d2[0]) / 2;
		landmark[2 * i + 1] = (d1[1] + d2[1]) / 2;
	}
	return landmark;
}

boost::tuple<Mat,vector<int> > Landmarker::getPatch(Mat img, vector<int> bounding, Point2f pts, float padding,Size sz)
{
	//pts是patch中心的相对坐标
	//转换相对坐标pts到绝对坐标point
	Point2f point(
		bounding[0] + pts.x * (bounding[1] - bounding[0]),
		bounding[2] + pts.y * (bounding[3] - bounding[2])
	);
	//计算patch的上下左右坐标
	vector<int> patch_bounding(4);
	patch_bounding[0] = point.x - (bounding[1] - bounding[0]) * padding;
	patch_bounding[1] = point.x + (bounding[1] - bounding[0]) * padding;
	patch_bounding[2] = point.y - (bounding[3] - bounding[2]) * padding;
	patch_bounding[3] = point.y + (bounding[3] - bounding[2]) * padding;
	//获取patch区域的图像
	Mat patch = crop(img,patch_bounding,sz);
	
	return boost::make_tuple(patch,patch_bounding);
}

Mat Landmarker::crop(Mat img, vector<int> bounding, Size size)
{
#if 1
    ublas::matrix<float> A(3,4),B(3,4);
    A(0,0) = bounding[0];	A(0,1) = bounding[0];		A(0,2) = bounding[1] + 1;	A(0,3) = bounding[1] + 1;
    A(1,0) = bounding[2];	A(1,1) = bounding[3] + 1;	A(1,2) = bounding[3] + 1;	A(1,3) = bounding[2];
    A(2,0) = 1;			A(2,1) = 1;			A(2,2) = 1;			A(2,3) = 1;
    B(0,0) = 0;	B(0,1) = 0;		B(0,2) = size.width;	B(0,3) = size.width;
    B(1,0) = 0; B(1,1) = size.height;	B(1,2) = size.height;	B(1,3) = 0;
    B(2,0) = 1; B(2,1) = 1;		B(2,2) = 1;		B(2,2) = 1;
    ublas::matrix<float> AAt = prod(A,trans(A));
    ublas::matrix<float> ABt = prod(A,trans(B));
    ublas::matrix<float> AAt_inv;
    svd_inv(AAt,AAt_inv);
    ublas::matrix<float> tmp = prod(AAt_inv,ABt);
    tmp = trans(tmp);
    Mat affine(Size(3,2),CV_32FC1,tmp.data().begin());
#else
    vector<Point> A, B;
    A.push_back(Point(bounding[0], bounding[2] ));
    A.push_back(Point(bounding[0], bounding[3] + 1));
    A.push_back(Point(bounding[1] + 1, bounding[3] + 1));
    A.push_back(Point(bounding[1] + 1, bounding[2] ));
    B.push_back(Point(0, 0));
    B.push_back(Point(0, size.height));
    B.push_back(Point(size.width, size.height));
    B.push_back(Point(size.width, 0));
    Mat affine = estimateRigidTransform(A, B, false);
#endif
    Mat patch;
    warpAffine(img, patch, affine, size);
    return patch;
}

vector<float> Landmarker::project(vector<int> bounding,vector<float> pos)
{
	//绝对坐标[x1 y1 x2 y2 ... xn yn]转换到相对坐标
	assert(pos.size() % 2 == 0);
	vector<float> retVal;
	for(int i = 0 ; i < pos.size() / 2 ; i++) {
		float x = pos[2 * i];
		float y = pos[2 * i + 1];
		retVal.push_back(static_cast<float>(x - bounding[0]) / (bounding[1] - bounding[0]));
		retVal.push_back(static_cast<float>(y - bounding[2]) / (bounding[3] - bounding[2]));
	}
#ifndef NDEBUG
	assert(pos.size() == retVal.size());
#endif
	return retVal;
}

vector<float> Landmarker::reproject(vector<int> bounding,vector<float> pos)
{
	//相对坐标[x1 y1 x2 y2 ... xn yn]转换到绝对坐标
	assert(pos.size() % 2 == 0);
	vector<float> retVal;
	for(int i = 0 ; i < pos.size() / 2 ; i++) {
		float x = pos[2 * i];
		float y = pos[2 * i + 1];
		retVal.push_back(bounding[0] + x * (bounding[1] - bounding[0]));
		retVal.push_back(bounding[2] + y * (bounding[3] - bounding[2]));
	}
#ifndef NDEBUG
	assert(pos.size() == retVal.size());
#endif
	return retVal;
}
