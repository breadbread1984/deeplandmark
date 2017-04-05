CAFFE_PREFIX=/home/xieyi/opt/caffe
CUDA_PREFIX=/usr
#CUDA_PREFIX=/home/sys411/opt/cuda-7.5
CXXFLAGS=-I. `pkg-config --cflags opencv` -I${CAFFE_PREFIX}/include -I${CAFFE_PREFIX}/.build_release/src/ -I${CUDA_PREFIX}/include -O2
LIBS=${CAFFE_PREFIX}/.build_release/src/caffe/proto/caffe.pb.o -L${CAFFE_PREFIX}/build/lib -lcaffe -lglog \
-lprotobuf `pkg-config --libs opencv` -lboost_program_options -lboost_filesystem -lboost_system \
-lboost_serialization -lboost_regex -pthread -llapack
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: generate_training_samples generate_solvers move_training_results show_dataset

generate_training_samples: generate_training_samples.o
	$(CXX) $^ $(LIBS) -o ${@}
	
generate_solvers: generate_solvers.o
	$(CXX) $^ $(LIBS) -o ${@}
	
move_training_results: move_training_results.o
	$(CXX) $^ $(LIBS) -o ${@}
	
show_dataset: show_dataset.o
	$(CXX) $^ $(LIBS) -o ${@}
	
clean:
	$(RM) $(OBJS) generate_training_samples generate_solvers move_training_results show_dataset
