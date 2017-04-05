CAFFE_PREFIX=/home/xieyi/opt/caffe/
CUDA_PREFIX=/usr
LIBS=-L${CAFFE_PREFIX}/build/lib ${CAFFE_PREFIX}/.build_release/src/caffe/proto/caffe.pb.o -lcaffe -lglog -lprotobuf `pkg-config --libs opencv` -lboost_filesystem -lboost_system -lboost_serialization -lboost_program_options -lboost_thread -pthread -llapack -fPIC
CXXFLAGS=-Iinclude -I${CAFFE_PREFIX}/include -I${CAFFE_PREFIX}/.build_release/src/ -I${CUDA_PREFIX}/include `pkg-config --cflags opencv` -msse3 -O2
OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

all: landmarker batch_landmarker

landmarker: src/main.o src/Landmarker.o src/Regressor.o
	$(CXX) $^ $(LIBS) -o ${@}
	
batch_landmarker: src/batch_landmarker.o src/Landmarker.o src/Regressor.o
	$(CXX) $^ $(LIBS) -o ${@}
	
clean:
	$(RM) $(OBJS) landmarker batch_landmarker

.PHONY: run
run:
	LD_LIBRARY_PATH=caffe/build/lib:${LD_LIBRARY_PATH} ./landmarker
