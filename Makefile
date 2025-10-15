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

$(OUTPUT_GPU): kernel_skeleton.cu $(COMMON_SRCS)
	# $(NVCC) $(CXXFLAGS) -lineinfo -dlto -arch=native -o $@ $^
	nvcc -O3 -lineinfo -o matcher kernel_skeleton.cu common.cc

$(OUTPUT_GEN): gen_sample.cc 
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OUTPUT_GEN_SIG): gen_sig.cc 
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(OUTPUT_GEN) $(OUTPUT_GEN_SIG) $(OUTPUT_GPU)