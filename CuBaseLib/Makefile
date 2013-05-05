
include ../tnet.mk

INCLUDE = -I. -I../ -I../KaldiLib 


CUDA_INCLUDE= -I$(CUDA_TK_BASE)/include
CUDA_BIN=$(CUDA_TK_BASE)/bin


CUSRC=$(wildcard *.cu)
CUOBJ=$(patsubst %.cu, %.o, $(CUSRC))



CUDA_FLAGS = -g -Xcompiler -fPIC --verbose 
ifeq ($(BITS64), true)
  CUDA_FLAGS += --machine 64
  BUT_FORCE_GCC64  = ln -s `which x86_64-linux-gcc` $(PWD)/gcc 
  BUT_UNLINK_GCC64 = unlink $(PWD)/gcc
else
  CUDA_FLAGS += --machine 32
endif


#Default compute capability architectures we compile with
CUDA_ARCH=-gencode arch=compute_10,code=sm_10 \
          -gencode arch=compute_13,code=sm_13 \
	  -gencode arch=compute_20,code=sm_20
#Get the CUDA Toolkit version (remove decimal point char)
CUDA_VERSION=$(shell $(CUDA_BIN)/nvcc -V | grep release | sed -e 's|.*release ||' -e 's|,.*||' -e 's|\.||')
#For toolkit 4.2 or newer, add the compute capability 3.0 
CUDA_VER_GT_4_2 := $(shell [ $(CUDA_VERSION) -ge 42 ] && echo true)
ifeq ($(CUDA_VER_GT_4_2), true)
  CUDA_ARCH += -gencode arch=compute_30,code=sm_30
endif




all : libCuBase.a

libCuBase.a : $(CUOBJ) $(OBJ)
	$(AR) ruv $@ $?
	$(RANLIB) $@


%.o : %.cu
	$(BUT_FORCE_GCC64)
	export PATH=$(PWD):$(CUDA_BIN):$(PATH); $(CUDA_BIN)/nvcc -c $< -o $@ -I. $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH)
	$(BUT_UNLINK_GCC64)

%.o : %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(CUDA_INCLUDE) $(INCLUDE)




.PHONY: clean depend

clean :
	rm -f *.o *.a

depend:
	$(CXX) -M $(CXXFLAGS) *.cc $(INCLUDE) $(CUDA_INCLUDE) > .depend.mk

-include .depend.mk

