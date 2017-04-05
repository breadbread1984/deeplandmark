#include <cstdlib>
#include <iostream>
#include <map>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::filesystem;

void foreach_dir(path dir);

int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&inputdir),"输入路径");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help") || 0 == vm.count("input")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	path input(inputdir);
	if(false == exists(input)||false == is_directory(input)) {
		cout<<"输入路径不正确！"<<endl;
		return EXIT_FAILURE;
	}
	
	for(directory_iterator itr(input) ; itr != directory_iterator() ; itr++)
		if(is_directory(*itr)) foreach_dir(*itr);
		
	return EXIT_SUCCESS;
}

void foreach_dir(path dir)
{
	//dir是某个模型的全部训练文件
	string dirname = dir.filename().string();
	regex expression(dirname + "_iter_([0-9]+)\\.caffemodel");
	map<int,path> s;
	for(directory_iterator itr(dir) ; itr != directory_iterator() ; itr++) {
		string filename = itr->path().filename().string();
		cmatch what;
		if(regex_match(filename.c_str(),what,expression)) {
			string id(what[1].first,what[1].second);
			s.insert(make_pair(lexical_cast<int>(id),itr->path()));
		} else continue;
	}
	if(s.size()) {
		remove_all(dir.parent_path() / (dirname + ".caffemodel"));
		copy_file(s.rbegin()->second,dir.parent_path() / (dirname + ".caffemodel"));
	}
}
