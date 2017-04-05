#include <cstdlib>
#include <iostream>
#include <functional>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <opencv2/opencv.hpp>
#include "Landmarker.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::program_options;
using namespace boost::archive;
using namespace boost::serialization;
using namespace cv;

#define THREAD_NUM 8

void marker(path dir,map<string,vector<Point> > & landmarks,mutex & lock);
Rect chooseOne(vector<boost::tuple<Rect,float> > faces);

int main(int argc,char ** argv)
{
	options_description desc;
	string input,output;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&input),"CASIA-WebFace的目录")
		("output,o",value<string>(&output),"输出标注文件路径");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	if(1 == argc || vm.count("help") || 1 != vm.count("input") || 1 != vm.count("output")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	path inputdir(input);
	if(false == exists(inputdir) || false == is_directory(inputdir)) {
		cout<<"输入路径错误"<<endl;
		return EXIT_FAILURE;
	}
	map<string,vector<Point> > landmarks;
	mutex lock;
	vector<boost::shared_ptr<thread> > handlers;
	for(directory_iterator itr(inputdir) ; itr != directory_iterator() ; itr++) {
		if(handlers.size() < THREAD_NUM) {
			handlers.push_back(boost::shared_ptr<thread> (
				new thread(marker,itr->path(),boost::ref(landmarks),boost::ref(lock))
			));
		}
		if(handlers.size() >= THREAD_NUM) {
			for(int i = 0 ; i < handlers.size() ; i++) handlers[i]->join();
			handlers.clear();
		}
	}
	if(handlers.size()) {
		for(int i = 0 ; i < handlers.size() ; i++) handlers[i]->join();
		handlers.clear();
	}
	path outputfile(output);
	remove_all(outputfile);
	std::ofstream out(output.c_str());
	text_oarchive oa(out);
	oa << landmarks;
}

void marker(path dir,map<string,vector<Point> > & landmarks,mutex & lock)
{
	Landmarker lm;
	CascadeClassifier detector("model_values/haarcascade_frontalface_default.xml");
	map<string,vector<Point> > data;
	for(directory_iterator itr(dir) ; itr != directory_iterator() ; itr++) {
		path file = itr->path();
		if(file.extension() == ".jpg" || file.extension() == ".png") {
			Mat img = imread(file.string());
			vector<Rect> faces;
			detector.detectMultiScale(img,faces);
			Rect face = faces[0];
			vector<Point> keypoints = lm.detectLandmark(img,face);
			data.insert(make_pair(dir.filename().string() + "/" + file.filename().string(),keypoints));
		}
	}
	{
		mutex::scoped_lock lockThisScope(lock);
		landmarks.insert(data.begin(),data.end());
	}
}

class Greater : public binary_function<bool,boost::tuple<Rect,float>,boost::tuple<Rect,float> > {
public:
	bool operator()(const boost::tuple<Rect,float> & a, const boost::tuple<Rect,float> & b) {
		get<0>(a).width * get<0>(a).height * get<1>(a) > get<0>(b).width * get<0>(b).height * get<1>(b);
	}
};

namespace boost {
	namespace serialization {
		template<class Archive> void serialize(Archive & ar, Point & p, const unsigned int version) {
			ar & p.x & p.y;
		}
	}
}
