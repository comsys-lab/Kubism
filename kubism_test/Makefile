NVCC := nvcc
CPP  := g++

CXXFLAGS := -std=c++11
NVCCFLAGS := -std=c++11

# CUDA 라이브러리 경로 설정
LDFLAGS := -L/usr/local/cuda/lib64 -lcudart

SRCS := main.cu cpu_function.cpp
OBJS := main.o cpu_function.o gpu_function.o

TARGET := main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CPP) $(OBJS) -o $(TARGET) $(LDFLAGS)

main.o: main.cu
	$(NVCC) $(NVCCFLAGS) -c main.cu -o main.o

cpu_function.o: cpu_function.cpp
	$(CPP) $(CXXFLAGS) -c cpu_function.cpp -o cpu_function.o

gpu_function.o: gpu_function.cu
	$(NVCC) $(NVCCFLAGS) -c gpu_function.cu -o gpu_function.o

clean:
	rm -f $(OBJS) $(TARGET)
