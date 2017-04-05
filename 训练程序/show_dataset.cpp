#include <cstdlib>
#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <caffe/util/format.hpp>
#include <caffe/util/rng.hpp>
#include <caffe/util/db.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace cv;
using namespace caffe;

int main(int argc,char ** argv)
{
	if(argc != 2) {
		cout<<"Usage: "<<argv[0]<<" <lmdb>"<<endl;
		return EXIT_SUCCESS;
	}
	path rootdir(argv[1]);
	if(
		false == exists(rootdir) || false == is_directory(rootdir) ||
		false == exists(rootdir / "img") || false == exists(rootdir / "landmark")
	) {
		cout<<"invalid directory"<<endl;
		return EXIT_FAILURE;
	}
	
	scoped_ptr<db::DB> imgdb(db::GetDB("lmdb"));
	imgdb->Open((rootdir / "img").string(),db::READ);
	scoped_ptr<db::Cursor> imgcursor(imgdb->NewCursor());
	scoped_ptr<db::DB> landmarkdb(db::GetDB("lmdb"));
	landmarkdb->Open((rootdir / "landmark").string(),db::READ);
	scoped_ptr<db::Cursor> landmarkcursor(landmarkdb->NewCursor());
	
	Datum datum;
	namedWindow("debug",0);
	do {
		datum.ParseFromString(imgcursor->value());
		//图片
		Mat img;
		if(datum.channels() > 1)
			img = Mat(Size(datum.width(),datum.height()),CV_8UC3);
		else
			img = Mat(Size(datum.width(),datum.height()),CV_8UC1);
		const string & data = datum.data();
		int index = 0;
		for(int c = 0 ; c < datum.channels() ; c++)
			for(int h = 0 ; h < datum.height() ; h++)
				for(int w = 0 ; w < datum.width() ; w++) {
					img.ptr(h)[datum.channels() * w + c] = data[index++];
				}
		datum.ParseFromString(landmarkcursor->value());
		//标注
		vector<float> landmark;
		//向量
		index = 0;
		for(int c = 0 ; c < datum.channels() ; c++)
			landmark.push_back(datum.float_data(index++));
		//显示结果
		for(int i = 0 ; i < landmark.size() / 2 ; i++)
			circle(img,Point(landmark[i*2]*img.cols,landmark[i*2+1]*img.rows),1,Scalar(0,0,255));
		imshow("debug",img);
		waitKey();
		imgcursor->Next();
		landmarkcursor->Next();
	} while(imgcursor->valid());
	
	return EXIT_SUCCESS;
}
