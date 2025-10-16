# Makefile

NVCC        := nvcc
COMMON_SRCS := common.cc
HEADERS := kseq/kseq.h

CXXFLAGS := -std=c++20 -O3

OUTPUT_GEN  := gen_sample
OUTPUT_GEN_SIG  := gen_sig
OUTPUT_GPU  := matcher

.DEFAULT_TARGET: all

all: $(OUTPUT_GEN) $(OUTPUT_GEN_SIG) $(OUTPUT_GPU)

$(OUTPUT_GPU): kernel_skeleton.o common.o
	$(NVCC) -std=c++17 -O3 -lineinfo -gencode=arch=compute_86,code=sm_86 -o $(OUTPUT_GPU) kernel_skeleton.o common.o

kernel_skeleton.o: kernel_skeleton.cu
	$(NVCC) -std=c++17 -O3 -lineinfo -gencode=arch=compute_86,code=sm_86 -c -o $@ $<

common.o: common.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OUTPUT_GEN): gen_sample.cc 
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OUTPUT_GEN_SIG): gen_sig.cc 
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OUTPUT_GEN) $(OUTPUT_GEN_SIG) $(OUTPUT_GPU) kernel_skeleton.o common.o