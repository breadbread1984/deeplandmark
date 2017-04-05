#include <cstdlib>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/program_options.hpp>
#include <boost/tuple/tuple.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::archive;
using namespace boost::program_options;

int main(int argc,char ** argv)
{
	options_description desc;
	string trainlist,testlist,outputdir;
	unsigned int batch_size;
	desc.add_options()
		("help,h","打印当前使用方法")
		("train,i",value<string>(&trainlist),"训练集合列表")
		("test,j",value<string>(&testlist),"测试集合列表")
		("output,o",value<string>(&outputdir),"输出路径")
		("batchsize,b",value<unsigned int>(&batch_size),"批处理大小");
	
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 0 == vm.count("train") || 0 == vm.count("test") || 0 == vm.count("output") || 0 == vm.count("batchsize")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	std::ifstream trainlistfile(trainlist.c_str());
	std::ifstream testlistfile(testlist.c_str());
	if(false == trainlistfile.is_open() || false == testlistfile.is_open()) {
		cout<<"训练样本列表或测试样本列表文件出错！"<<endl;
		return EXIT_FAILURE;
	}
	
	path outputroot(outputdir);
	remove_all(outputroot);
	create_directory(outputroot);
	
	text_iarchive trlf(trainlistfile);
	text_iarchive telf(testlistfile);
	
	map<string,int> trainset_nums,testset_nums;
	trlf >> trainset_nums;
	telf >> testset_nums;
	
	for(map<string,int>::iterator it = trainset_nums.begin() ; it != trainset_nums.end() ; it++) {
		//训练集和测试集都需要有
		map<string,int>::iterator it2 = testset_nums.find(it->first);
		if(testset_nums.end() == it2) {
			cout<<it->first<<"在testset_nums中找不到"<<endl;
			return EXIT_FAILURE;
		}
		//输出路径
		path filepath = outputroot / (it->first + "_solver.prototxt");
		std::ofstream out(filepath.string().c_str());
#ifndef NDEBUG
		assert(out.is_open());
#endif
		int epoch = ceil(static_cast<float>(it->second) / batch_size);
		out<<"net: \""<<it->first + "_train.prototxt\""<<endl;
		out<<"test_iter: "<<ceil(static_cast<float>(it2->second) / batch_size)<<endl;
		out<<"test_interval: "<<epoch<<endl;
		out<<"momentum: 0.9"<<endl;
		out<<"weight_decay: 0.0005"<<endl;
		out<<"lr_policy: \"inv\""<<endl;
		out<<"base_lr: 0.01"<<endl;
		out<<"gamma: 0.0001"<<endl;
		out<<"power: 0.75"<<endl;
		out<<"max_iter: "<<5000 * epoch<<endl;
		out<<"display: "<<200<<endl;
		out<<"snapshot: "<<50 * epoch<<endl;
		out<<"snapshot_prefix: \"../model_values/"<<it->first<<"/"<<it->first<<"\""<<endl;
		out<<"solver_mode: GPU"<<endl;
	}
	
	return EXIT_SUCCESS;
}
