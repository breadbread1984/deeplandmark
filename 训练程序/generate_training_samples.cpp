#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/math/constants/constants.hpp>
#include <opencv2/opencv.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <caffe/util/format.hpp>
#include <caffe/util/rng.hpp>
#include <caffe/util/db.hpp>
#include "matrix_basic.hpp"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;
using namespace boost::serialization;
using namespace boost::archive;
using namespace cv;
using namespace caffe;

vector<boost::tuple<string,vector<int>,vector<Point2f> > > parseList(string file); 
boost::tuple<Mat,vector<Point2f> > rotate(Mat face, vector<int> bounding, vector<Point2f> pts, double alpha);
boost::tuple<Mat,vector<Point2f> > flip(Mat face, vector<Point2f> pts);
vector<Point2f> randomShift(vector<Point2f> pts, float shift);
vector<vector<Point2f> > randomShiftWithArgument(vector<Point2f> pts, float shift);
vector<Point2f> project(vector<int> bounding, vector<Point2f> pts);
Point2f project(vector<int> bounding, Point2f pts);
vector<Point2f> reproject(vector<int> bounding, vector<Point2f> pts);
Point2f reproject(vector<int> bounding, Point2f pts);
boost::tuple<Mat,vector<int> > getPatch(Mat img, vector<int> bounding, Point2f pts, float padding,Size sz);
Mat crop(Mat img, vector<int> bounding, Size size);
void write2lmdb(vector<boost::tuple<string,vector<Point2f> > > & list, boost::shared_ptr<db::DB> img_db, boost::shared_ptr<db::DB> landmark_db, boost::shared_ptr<db::Transaction> img_txn, boost::shared_ptr<db::Transaction> landmark_txn);
void write2lmdb(vector<boost::tuple<string,Point2f> > & list, boost::shared_ptr<db::DB> img_db, boost::shared_ptr<db::DB> landmark_db, boost::shared_ptr<db::Transaction> img_txn, boost::shared_ptr<db::Transaction> landmark_txn);
void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);
void vectorToDatum(const vector<float> & v,Datum * datum);
map<string,int> generate_level1(vector<boost::tuple<string,vector<int>,vector<Point2f> > > & list,path outputdir);
map<string,int> generate_level2(vector<boost::tuple<string,vector<int>,vector<Point2f> > > & list,path outputdir);
map<string,int> generate_level3(vector<boost::tuple<string,vector<int>,vector<Point2f> > > & list, path outputdir);

static mt19937 BaseGenerator;
static variate_generator<mt19937&,normal_distribution<double> > normRandGen(BaseGenerator,normal_distribution<double>(0,1));

int main(int argc,char ** argv)
{
	options_description desc;
	string listfile,outputdir;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&listfile),"输入列表文件")
		("output,o",value<string>(&outputdir),"输出路径");
	
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 0 == vm.count("input") || 0 == vm.count("output")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
		
	//读取列表文件
	vector<boost::tuple<string,vector<int>,vector<Point2f> > > list = parseList(listfile);
	
	map<string,int> retVal;
	map<string,int> dataset_nums;
	
	path outputroot(outputdir);
	remove_all(outputroot);
	create_directory(outputroot);
	
	retVal = generate_level1(list,path(outputdir) / "level1"); dataset_nums.insert(retVal.begin(),retVal.end());
	retVal = generate_level2(list,path(outputdir) / "level2"); dataset_nums.insert(retVal.begin(),retVal.end());
	retVal = generate_level3(list,path(outputdir) / "level3"); dataset_nums.insert(retVal.begin(),retVal.end());
	//输出dat格式的样本数量文件
	std::ofstream dat("dataset_nums.dat");
	text_oarchive oa(dat);
	oa << dataset_nums;
	
	return EXIT_SUCCESS;
}

map<string,int> generate_level1(vector<boost::tuple<string,vector<int>,vector<Point2f> > > & list,path outputdir)
{
	path outputroot("tmp");
	remove_all(outputroot);
	create_directory(outputroot);
	
	vector<boost::tuple<string,vector<Point2f> > > facelist;
	vector<boost::tuple<string,vector<Point2f> > > eyesnoselist;
	vector<boost::tuple<string,vector<Point2f> > > nosemouthlist;
	for(vector<boost::tuple<string,vector<int>,vector<Point2f> > >::iterator it = list.begin() ; it != list.end() ; it++) {
		path imgpath(get<0>(*it));
		string basename = imgpath.stem().string();
		string extension = imgpath.extension().string();
		
		Mat clr = imread(get<0>(*it));
		if(true == clr.empty()) {
			cout<<get<0>(*it)<<" cant be opened!"<<endl;
			continue;
		}
		float left = get<1>(*it)[0]/* * 0.95*/;
		float right = get<1>(*it)[1]/* * 1.05*/;
		float top = get<1>(*it)[2]/* * 0.95*/;
		float bottom = get<1>(*it)[3]/* * 1.05*/;
		Rect bounding(left,top,right - left + 1,bottom - top + 1);
		//原始灰度版本和翻转灰度版本
		Mat gray;
		cvtColor(clr,gray,CV_BGR2GRAY);
		//整脸图像样本
		Mat face = gray(bounding);
		Mat face_resize;
		resize(face,face_resize,Size(39,39));
		path outputpath = outputroot / ("face_" + basename + extension);
		imwrite(outputpath.string(),face_resize);
		facelist.push_back(boost::make_tuple(outputpath.string(),get<2>(*it)));
		if(normRandGen() > -1) {
			//图像水平翻转
			boost::tuple<Mat,vector<Point2f> > flipped = flip(face,get<2>(*it));
			Mat face_flipped;
			resize(get<0>(flipped),face_flipped,Size(39,39));
			//输出图像并且记录关键点
			path outputpath = outputroot / ("face_" + basename + "_flip" + extension);
			imwrite(outputpath.string(),face_flipped);
			facelist.push_back(boost::make_tuple(outputpath.string(),get<1>(flipped)));
		}
		if(normRandGen() > 0.5) {
			//1）图像旋转
			boost::tuple<Mat,vector<Point2f> > rotated = rotate(gray,get<1>(*it),reproject(get<1>(*it),get<2>(*it)),5);
			//变成相对坐标
			vector<Point2f> landmark_rotated = project(get<1>(*it),get<1>(rotated));
			Mat face_rotated_by_alpha;
			resize(get<0>(rotated),face_rotated_by_alpha,Size(39,39));
			//输出图像并且记录关键点
			path outputpath = outputroot / ("face_" + basename + "_rotate+5" + extension);
			imwrite(outputpath.string(),face_rotated_by_alpha);
			facelist.push_back(boost::make_tuple(outputpath.string(),landmark_rotated));
			//2）图像水平翻转
			boost::tuple<Mat,vector<Point2f> > flipped = flip(get<0>(rotated),landmark_rotated);
			Mat face_flipped;
			resize(get<0>(flipped),face_flipped,Size(39,39));
			//输出图像并且记录关键点
			outputpath = outputroot / ("face_" + basename + "_rotate+5flip" + extension);
			imwrite(outputpath.string(),face_flipped);
			facelist.push_back(boost::make_tuple(outputpath.string(),get<1>(flipped)));
		}
		if(normRandGen() > 0.5) {
			//1）图像旋转
			boost::tuple<Mat,vector<Point2f> > rotated = rotate(gray,get<1>(*it),reproject(get<1>(*it),get<2>(*it)),-5);
			//变成相对坐标
			vector<Point2f> landmark_rotated = project(get<1>(*it),get<1>(rotated));
			Mat face_rotated_by_alpha;
			resize(get<0>(rotated),face_rotated_by_alpha,Size(39,39));
			//输出图像并且记录关键点
			path outputpath = outputroot / ("face_" + basename + "_rotate-5" + extension);
			imwrite(outputpath.string(),face_rotated_by_alpha);
			facelist.push_back(boost::make_tuple(outputpath.string(),landmark_rotated));
			//2）图像水平翻转
			boost::tuple<Mat,vector<Point2f> > flipped = flip(get<0>(rotated),landmark_rotated);
			Mat face_flipped;
			resize(get<0>(flipped),face_flipped,Size(39,39));
			//输出图像并且记录关键点
			outputpath = outputroot / ("face_" + basename + "_rotate-5flip" + extension);
			imwrite(outputpath.string(),face_flipped);
			facelist.push_back(boost::make_tuple(outputpath.string(),get<1>(flipped)));
		}
		//眼睛鼻子图片样本
		Mat eyesnose = face_resize(Rect(0,0,39,31));
		outputpath = outputroot / ("eyesnose_" + basename + extension);
		imwrite(outputpath.string(),eyesnose);
		vector<Point2f> en_landmark(get<2>(*it).begin(),get<2>(*it).begin() + 3);
		//眼睛鼻子关键点重新映射到眼睛鼻子区域相对坐标
		vector<int> eyesnose_bounding(4);
		eyesnose_bounding[0] = get<1>(*it)[0]; //left
		eyesnose_bounding[1] = get<1>(*it)[1]; //right
		eyesnose_bounding[2] = get<1>(*it)[2]; //top
		eyesnose_bounding[3] = get<1>(*it)[2] + (31.0 / 39) * (get<1>(*it)[3] - get<1>(*it)[2]); //bottom
		en_landmark = project(eyesnose_bounding, reproject(get<1>(*it),en_landmark));
		eyesnoselist.push_back(boost::make_tuple(outputpath.string(),en_landmark));
		if(normRandGen() > 0.5) {
			//图像水平翻转
			boost::tuple<Mat,vector<Point2f> > flipped = flip(eyesnose,get<2>(*it));
			vector<Point2f> en_landmark(get<1>(flipped).begin(),get<1>(flipped).begin() + 3);
			en_landmark = project(eyesnose_bounding, reproject(get<1>(*it),en_landmark));			
			outputpath = outputroot / ("eyesnose_" + basename + "_flip" + extension);
			imwrite(outputpath.string(),get<0>(flipped));
			eyesnoselist.push_back(boost::make_tuple(outputpath.string(),en_landmark));
		}
		//鼻子嘴巴图片样本
		Mat nosemouth = face_resize(Rect(0,8,39,31));
		outputpath = outputroot / ("nosemouth_" + basename + extension);
		imwrite(outputpath.string(),nosemouth);
		vector<Point2f> nm_landmark(get<2>(*it).begin() + 2,get<2>(*it).begin() + 5);
		vector<int> nosemouth_bounding(4);
		nosemouth_bounding[0] = get<1>(*it)[0];
		nosemouth_bounding[1] = get<1>(*it)[1];
		nosemouth_bounding[2] = get<1>(*it)[2] + (8.0 / 39) * (get<1>(*it)[3] - get<1>(*it)[2]);
		nosemouth_bounding[3] = get<1>(*it)[3];
		nm_landmark = project(nosemouth_bounding,reproject(get<1>(*it),nm_landmark));
		nosemouthlist.push_back(boost::make_tuple(outputpath.string(),nm_landmark));
		if(normRandGen() > 0.5) {
			//图像水平翻转
			boost::tuple<Mat,vector<Point2f> > flipped = flip(nosemouth,get<2>(*it));
			vector<Point2f> nm_landmark(get<1>(flipped).begin() + 2,get<1>(flipped).begin() + 5);
			nm_landmark = project(nosemouth_bounding,reproject(get<1>(*it),nm_landmark));
			outputpath = outputroot / ("nosemouth_" + basename + "_flip" + extension);
			imwrite(outputpath.string(),get<0>(flipped));
			nosemouthlist.push_back(boost::make_tuple(outputpath.string(),nm_landmark));
		}
	}
	random_shuffle(facelist.begin(),facelist.end());
	random_shuffle(eyesnoselist.begin(),eyesnoselist.end());
	random_shuffle(nosemouthlist.begin(),nosemouthlist.end());

	//写入lmdb库
	outputroot = path(outputdir);
	remove_all(outputroot);
	create_directory(outputroot);
	create_directory(outputroot / "1_F");
	create_directory(outputroot / "1_EN");
	create_directory(outputroot / "1_NM");
	//1_F的图片库
	boost::shared_ptr<db::DB> _1f_img_db(db::GetDB("lmdb"));
	_1f_img_db->Open((outputroot / "1_F" / "img").string(),db::NEW);
	boost::shared_ptr<db::Transaction> _1f_img_txn(_1f_img_db->NewTransaction());
	//1_F的标注库
	boost::shared_ptr<db::DB> _1f_landmark_db(db::GetDB("lmdb"));
	_1f_landmark_db->Open((outputroot / "1_F" / "landmark").string(),db::NEW);
	boost::shared_ptr<db::Transaction> _1f_landmark_txn(_1f_landmark_db->NewTransaction());
	//1_EN的图像库
	boost::shared_ptr<db::DB> _1en_img_db(db::GetDB("lmdb"));
	_1en_img_db->Open((outputroot / "1_EN" / "img").string(),db::NEW);
	boost::shared_ptr<db::Transaction> _1en_img_txn(_1en_img_db->NewTransaction());
	//1_EN的标注库
	boost::shared_ptr<db::DB> _1en_landmark_db(db::GetDB("lmdb"));
	_1en_landmark_db->Open((outputroot / "1_EN" / "landmark").string(),db::NEW);
	boost::shared_ptr<db::Transaction> _1en_landmark_txn(_1en_landmark_db->NewTransaction());
	//1_NM的图像库
	boost::shared_ptr<db::DB> _1nm_img_db(db::GetDB("lmdb"));
	_1nm_img_db->Open((outputroot / "1_NM" / "img").string(),db::NEW);
	boost::shared_ptr<db::Transaction> _1nm_img_txn(_1nm_img_db->NewTransaction());
	//1_NM的标注库
	boost::shared_ptr<db::DB> _1nm_landmark_db(db::GetDB("lmdb"));
	_1nm_landmark_db->Open((outputroot / "1_NM" / "landmark").string(),db::NEW);
	boost::shared_ptr<db::Transaction> _1nm_landmark_txn(_1nm_landmark_db->NewTransaction());
	
	write2lmdb(facelist,_1f_img_db,_1f_landmark_db,_1f_img_txn,_1f_landmark_txn);
	write2lmdb(eyesnoselist,_1en_img_db,_1en_landmark_db,_1en_img_txn,_1en_landmark_txn);
	write2lmdb(nosemouthlist,_1nm_img_db,_1nm_landmark_db,_1nm_img_txn,_1nm_landmark_txn);
	
	remove_all("tmp");
	
	map<string,int> retVal;
	retVal.insert(make_pair("1_F",facelist.size()));
	retVal.insert(make_pair("1_EN",eyesnoselist.size()));
	retVal.insert(make_pair("1_NM",nosemouthlist.size()));
	return retVal;
}

map<string,int> generate_level2(vector<boost::tuple<string,vector<int>,vector<Point2f> > > & list, path outputdir)
{
	path outputroot("tmp");
	remove_all(outputroot);
	create_directory(outputroot);
	
	static const boost::tuple<int,string,float> types[] = {
		boost::make_tuple(0, "LE1", 0.16),boost::make_tuple(0, "LE2", 0.18),
		boost::make_tuple(1, "RE1", 0.16),boost::make_tuple(1, "RE2", 0.18),
		boost::make_tuple(2, "N1", 0.16),boost::make_tuple(2, "N2", 0.18),
		boost::make_tuple(3, "LM1", 0.16),boost::make_tuple(3, "LM2", 0.18),
		boost::make_tuple(4, "RM1", 0.16),boost::make_tuple(4, "RM2", 0.18)
	};
	
	map<string,vector<boost::tuple<string,Point2f> > > patchlists;
	for(vector<boost::tuple<string,vector<int>,vector<Point2f> > >::iterator it = list.begin() ; it != list.end() ; it++) {
		//对每一个样本
		path imgpath(get<0>(*it));
		string filename = imgpath.filename().string();

		Mat gray = imread(get<0>(*it),CV_LOAD_IMAGE_GRAYSCALE);
		vector<vector<Point2f> > landmarkPs = randomShiftWithArgument(get<2>(*it),0.05);
		for(int i = 0 ; i < landmarkPs.size() ; i++) {
			//对每个偏移版本的5关键点
			vector<Point2f> & landmarkP = landmarkPs[i]; //五个关键点
			for(int j = 0 ; j < 10 ; j++) {
				//对每个关键点
				boost::tuple<Mat,vector<int> > patch_data = getPatch(gray,get<1>(*it),landmarkP[get<0>(types[j])],get<2>(types[j]),Size(15,15));
				path outputpath = outputroot / (lexical_cast<string>(i) + "_" + get<1>(types[j]) + "_" + filename);
				//计算关键点坐标相对带偏移的patch的相对坐标
				//如果patch没有偏移那么相对坐标为(0.5,0.5)
				Point2f deviate = project(get<1>(patch_data),reproject(get<1>(*it),get<2>(*it)[get<0>(types[j])]));
				imwrite(outputpath.string(),get<0>(patch_data));
				patchlists[get<1>(types[j])].push_back(boost::make_tuple(outputpath.string(),deviate));
			}
		}
	}
	//写入lmdb库
	outputroot = path(outputdir);
	remove_all(outputroot);
	create_directory(outputroot);
	map<string,int> retVal;
	for(int j = 0 ; j < 10 ; j++) {
		//对每个关键点分别生成lmdb
		create_directory(outputroot / ("2_" + get<1>(types[j])));
		//对样本重新排序
		random_shuffle(patchlists[get<1>(types[j])].begin(),patchlists[get<1>(types[j])].end());
		//图片库
		boost::shared_ptr<db::DB> img_db(db::GetDB("lmdb"));
		img_db->Open((outputroot / ("2_" + get<1>(types[j])) / "img").string(),db::NEW);
		boost::shared_ptr<db::Transaction> img_txn(img_db->NewTransaction());
		//标注库
		boost::shared_ptr<db::DB> landmark_db(db::GetDB("lmdb"));
		landmark_db->Open((outputroot / ("2_" + get<1>(types[j])) / "landmark").string(),db::NEW);
		boost::shared_ptr<db::Transaction> landmark_txn(landmark_db->NewTransaction());
		write2lmdb(patchlists[get<1>(types[j])],img_db,landmark_db,img_txn,landmark_txn);
		retVal.insert(make_pair("2_" + get<1>(types[j]),patchlists[get<1>(types[j])].size()));
	}
	
	remove_all("tmp");
	
	return retVal;
}

map<string,int> generate_level3(vector<boost::tuple<string,vector<int>,vector<Point2f> > > & list, path outputdir)
{
	path outputroot("tmp");
	remove_all(outputroot);
	create_directory(outputroot);

	static const boost::tuple<int,string,float> types[] = {
		boost::make_tuple(0, "LE1", 0.11),boost::make_tuple(0, "LE2", 0.12),
		boost::make_tuple(1, "RE1", 0.11),boost::make_tuple(1, "RE2", 0.12),
		boost::make_tuple(2, "N1", 0.11),boost::make_tuple(2, "N2", 0.12),
		boost::make_tuple(3, "LM1", 0.11),boost::make_tuple(3, "LM2", 0.12),
		boost::make_tuple(4, "RM1", 0.11),boost::make_tuple(4, "RM2", 0.12)
	};
	
	map<string,vector<boost::tuple<string,Point2f> > > patchlists;
	for(vector<boost::tuple<string,vector<int>,vector<Point2f> > >::iterator it = list.begin() ; it != list.end() ; it++) {
		//对每一个样本
		path imgpath(get<0>(*it));
		string filename = imgpath.filename().string();

		Mat gray = imread(get<0>(*it),CV_LOAD_IMAGE_GRAYSCALE);
		vector<vector<Point2f> > landmarkPs = randomShiftWithArgument(get<2>(*it),0.01);
		for(int i = 0 ; i < landmarkPs.size() ; i++) {
			//对每个偏移版本的5关键点
			vector<Point2f> & landmarkP = landmarkPs[i]; //五个关键点
			for(int j = 0 ; j < 10 ; j++) {
				//对每个关键点
				boost::tuple<Mat,vector<int> > patch_data = getPatch(gray,get<1>(*it),landmarkP[get<0>(types[j])],get<2>(types[j]),Size(15,15));
				path outputpath = outputroot / (lexical_cast<string>(i) + "_" + get<1>(types[j]) + "_" + filename);
				//计算关键点坐标相对带偏移的patch的相对坐标
				//如果patch没有偏移那么相对坐标为(0.5,0.5)
				Point2f deviate = project(get<1>(patch_data),reproject(get<1>(*it),get<2>(*it)[get<0>(types[j])]));
				imwrite(outputpath.string(),get<0>(patch_data));
				patchlists[get<1>(types[j])].push_back(boost::make_tuple(outputpath.string(),deviate));
			}
		}
	}
	//写入lmdb库
	outputroot = path(outputdir);
	remove_all(outputroot);
	create_directory(outputroot);
	map<string,int> retVal;
	for(int j = 0 ; j < 10 ; j++) {
		//对每个关键点分别生成lmdb
		create_directory(outputroot / ("3_" + get<1>(types[j])));
		//对样本重新排序
		random_shuffle(patchlists[get<1>(types[j])].begin(),patchlists[get<1>(types[j])].end());
		//图片库
		boost::shared_ptr<db::DB> img_db(db::GetDB("lmdb"));
		img_db->Open((outputroot / ("3_" + get<1>(types[j])) / "img").string(),db::NEW);
		boost::shared_ptr<db::Transaction> img_txn(img_db->NewTransaction());
		//标注库
		boost::shared_ptr<db::DB> landmark_db(db::GetDB("lmdb"));
		landmark_db->Open((outputroot / ("3_" + get<1>(types[j])) / "landmark").string(),db::NEW);
		boost::shared_ptr<db::Transaction> landmark_txn(landmark_db->NewTransaction());
		write2lmdb(patchlists[get<1>(types[j])],img_db,landmark_db,img_txn,landmark_txn);
		retVal.insert(make_pair("3_" + get<1>(types[j]),patchlists[get<1>(types[j])].size()));
	}

	remove_all("tmp");
	
	return retVal;
}

vector<boost::tuple<string,vector<int>,vector<Point2f> > > parseList(string file)
{
/*
	Generate data from txt file
	return [(img_path, bbox, landmark)]
	bbox: [left, right, top, bottom]
	landmark: [(x1, y1), (x2, y2), ...]

 */
	std::ifstream in(file.c_str());
	if(false == in.is_open()) throw runtime_error("invalid list file!");
	string line;
	char_separator<char> sep(" \t");
	typedef boost::tokenizer<char_separator<char> > tokenizer;
	vector<boost::tuple<string,vector<int>,vector<Point2f> > > retVal;
	while(false == in.eof()) {
		getline(in,line);
		trim(line);
		if(line == "") continue;
		tokenizer tokens(line,sep);
		tokenizer::iterator tok_iter = tokens.begin();
		string file = *(tok_iter++);
		//目标框的上下左右绝对坐标
		vector<int> bounding(4);
		for(int i = 0 ; i < 4 ; i++) bounding[i] = lexical_cast<int>(*(tok_iter++));
		//读取相对关键点绝对坐标，并转换成相对坐标
		vector<Point2f> pts(5);
		for(int i = 0 ; i < 5 ; i++) {
			float x = lexical_cast<float>(*(tok_iter++));
			float y = lexical_cast<float>(*(tok_iter++));
			Point2f pt;
			pt.x = static_cast<float>(x - bounding[0]) / (bounding[1] - bounding[0]); 
			pt.y = static_cast<float>(y - bounding[2]) / (bounding[3] - bounding[2]); 
#ifndef NDEBUG
			assert(0 <= pt.x && pt.x <= 1);
			assert(0 <= pt.y && pt.y <= 1);
#endif
			pts[i] = pt;
		}
		retVal.push_back(boost::make_tuple(file,bounding,pts));
	}
	return retVal;
}

boost::tuple<Mat,vector<Point2f> > rotate(Mat face, vector<int> bounding, vector<Point2f> pts, double alpha)
{
	//face是原始图像
	//pts是绝对坐标
	Point center((bounding[0] + bounding[1]) / 2, (bounding[2] + bounding[3]) / 2);
	Mat rot_mat = Mat::eye(Size(3,2),CV_32FC1);
	//getRotationMatrix2D(center,alpha,1);
	float rad = alpha / 180 * math::constants::pi<double>();
	rot_mat.at<float>(0,0) = cos(rad);
	rot_mat.at<float>(0,1) = sin(rad);
	rot_mat.at<float>(1,0) = -sin(rad);
	rot_mat.at<float>(1,1) = cos(rad);
	rot_mat.at<float>(0,2) = -(center.x * rot_mat.at<float>(0,0) + center.y * rot_mat.at<float>(0,1) - center.x);
	rot_mat.at<float>(1,2) = -(center.x * rot_mat.at<float>(1,0) + center.y * rot_mat.at<float>(1,1) - center.y);
	Mat img_rotated_by_alpha;
	warpAffine(face,img_rotated_by_alpha,rot_mat,face.size());
	vector<Point2f> rotpts(5);
	for(int i = 0 ; i < 5 ; i++)
		rotpts[i] = Point2f(
			rot_mat.at<float>(0,0) * pts[i].x + rot_mat.at<float>(0,1) * pts[i].y + rot_mat.at<float>(0,2),
			rot_mat.at<float>(1,0) * pts[i].x + rot_mat.at<float>(1,1) * pts[i].y + rot_mat.at<float>(1,2)
		);
	Mat rotface = img_rotated_by_alpha(Rect(bounding[0], bounding[2], bounding[1] - bounding[0] + 1, bounding[3] - bounding[2] + 1));
	//返回旋转后图片中框出来的人脸子图和旋转后的绝对坐标
	return boost::make_tuple(rotface,rotpts);
}

boost::tuple<Mat,vector<Point2f> > flip(Mat face, vector<Point2f> pts)
{
	//pts是相对坐标
	Mat flpface;
	//图像水平翻转
	flip(face,flpface,1);
	//修改关键点坐标
	vector<Point2f> flppts(5);
	flppts[0] = Point2f(1 - pts[1].x, pts[1].y);
	flppts[1] = Point2f(1 - pts[0].x, pts[0].y);
	flppts[2] = Point2f(1 - pts[2].x, pts[2].y);
	flppts[3] = Point2f(1 - pts[4].x, pts[4].y);
	flppts[4] = Point2f(1 - pts[3].x, pts[3].y);
	
	return boost::make_tuple(flpface,flppts);
}

vector<Point2f> randomShift(vector<Point2f> pts, float shift)
{
	//pts是相对坐标
	vector<Point2f> sftpts(5);
	for(int i = 0 ; i < 5 ; i++) {
		sftpts[i].x = pts[i].x + (2 * normRandGen() - 1) * shift;
		sftpts[i].y = pts[i].y + (2 * normRandGen() - 1) * shift;
	}
	return sftpts;
}

vector<vector<Point2f> > randomShiftWithArgument(vector<Point2f> pts, float shift)
{
	vector<vector<Point2f> > retVal;
	//pts是相对坐标
	const int N = 5;
	for(int i = 0 ; i < N ; i++)
		retVal.push_back(randomShift(pts,shift));
	return retVal;
}

vector<Point2f> project(vector<int> bounding, vector<Point2f> pts)
{
	//将绝对坐标变成相对坐标
	vector<Point2f> retVal(pts.size());
	for(int i = 0 ; i < pts.size() ; i++) {
		retVal[i].x = static_cast<float>(pts[i].x - bounding[0]) / (bounding[1] - bounding[0]);
		retVal[i].y = static_cast<float>(pts[i].y - bounding[2]) / (bounding[3] - bounding[2]);
	}
	return retVal;
}

Point2f project(vector<int> bounding, Point2f pts)
{
	//将绝对坐标变成相对坐标
	Point2f retVal;
	retVal.x = static_cast<float>(pts.x - bounding[0]) / (bounding[1] - bounding[0]);
	retVal.y = static_cast<float>(pts.y - bounding[2]) / (bounding[3] - bounding[2]);
	return retVal;
}

vector<Point2f> reproject(vector<int> bounding, vector<Point2f> pts)
{
	//将相对坐标变成绝对坐标
	vector<Point2f> retVal(pts.size());
	for(int i = 0 ; i < pts.size() ; i++) {
		retVal[i].x = bounding[0] + pts[i].x * (bounding[1] - bounding[0]);
		retVal[i].y = bounding[2] + pts[i].y * (bounding[3] - bounding[2]);
	}
	return retVal;
}

Point2f reproject(vector<int> bounding, Point2f pts)
{
	//将相对坐标变成绝对坐标
	Point2f retVal;
	retVal.x = bounding[0] + pts.x * (bounding[1] - bounding[0]);
	retVal.y = bounding[2] + pts.y * (bounding[3] - bounding[2]);
	return retVal;
}

boost::tuple<Mat,vector<int> > getPatch(Mat img, vector<int> bounding, Point2f pts, float padding,Size sz)
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

Mat crop(Mat img, vector<int> bounding, Size size)
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

void write2lmdb(
	vector<boost::tuple<string,vector<Point2f> > > & list, 
	boost::shared_ptr<db::DB> img_db, boost::shared_ptr<db::DB> landmark_db, 
	boost::shared_ptr<db::Transaction> img_txn, boost::shared_ptr<db::Transaction> landmark_txn
) {
	Datum datum;
	int count = 0;
	for(vector<boost::tuple<string,vector<Point2f> > >::iterator it = list.begin() ; it != list.end() ; it++) {
		//写入图像
		Mat img = imread(get<0>(*it),CV_LOAD_IMAGE_GRAYSCALE);
		if(true == img.empty()) {
			cout<<"图像"<<get<0>(*it)<<"无法打开！"<<endl;
			continue;
		}
		CVMatToDatum(img,&datum);
		string buffer;
		datum.SerializeToString(&buffer);
		img_txn->Put(lexical_cast<string>(count),buffer);
		//写入标注
		vector<float> values;
		for(vector<Point2f>::iterator itt = get<1>(*it).begin() ; itt != get<1>(*it).end() ; itt++) {
			values.push_back(itt->x);
			values.push_back(itt->y);
		}
		vectorToDatum(values,&datum);
		datum.SerializeToString(&buffer);
		landmark_txn->Put(lexical_cast<string>(count),buffer);
		++count;
		if(count%1000 == 0) {
			img_txn->Commit();
			img_txn.reset(img_db->NewTransaction());
			landmark_txn->Commit();
			landmark_txn.reset(landmark_db->NewTransaction());
		}
	}
	 if(count % 1000 != 0) {
		img_txn->Commit();
		landmark_txn->Commit();
	}
}

void write2lmdb(
	vector<boost::tuple<string,Point2f> > & list, 
	boost::shared_ptr<db::DB> img_db, boost::shared_ptr<db::DB> landmark_db, 
	boost::shared_ptr<db::Transaction> img_txn, boost::shared_ptr<db::Transaction> landmark_txn
) {
	Datum datum;
	int count = 0;
	for(vector<boost::tuple<string,Point2f> >::iterator it = list.begin() ; it != list.end() ; it++) {
		//写入图像
		Mat img = imread(get<0>(*it),CV_LOAD_IMAGE_GRAYSCALE);
		CVMatToDatum(img,&datum);
		string buffer;
		datum.SerializeToString(&buffer);
		img_txn->Put(lexical_cast<string>(count),buffer);
		//写入标注
		vector<float> values;
		values.push_back(get<1>(*it).x);
		values.push_back(get<1>(*it).y);
		vectorToDatum(values,&datum);
		datum.SerializeToString(&buffer);
		landmark_txn->Put(lexical_cast<string>(count),buffer);
		++count;
		if(count%1000 == 0) {
			img_txn->Commit();
			img_txn.reset(img_db->NewTransaction());
			landmark_txn->Commit();
			landmark_txn.reset(landmark_db->NewTransaction());
		}
	}
	 if(count % 1000 != 0) {
		img_txn->Commit();
		landmark_txn->Commit();
	}	
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) 
{
	CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
	datum->set_channels(cv_img.channels());
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);
	int datum_channels = datum->channels();
	int datum_height = datum->height();
	int datum_width = datum->width();
	int datum_size = datum_channels * datum_height * datum_width;
	std::string buffer(datum_size, ' ');
	for (int h = 0; h < datum_height; ++h) {
		const uchar* ptr = cv_img.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < datum_width; ++w)
			for (int c = 0; c < datum_channels; ++c) {
				int datum_index = (c * datum_height + h) * datum_width + w;
				buffer[datum_index] = static_cast<char>(ptr[img_index++]);
			}
	}
	datum->set_data(buffer);
}

void vectorToDatum(const vector<float> & v,Datum * datum)
{
	datum->set_channels(v.size());
	datum->set_height(1);
	datum->set_width(1);
	datum->clear_data();
	datum->clear_float_data();
	datum->set_encoded(false);
	for(int i = 0 ; i < v.size() ; i++) datum->add_float_data(v[i]);
}
