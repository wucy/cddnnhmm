
include tnet.mk

##### Check that CUDA Toolkit directory was set
ifneq ($(HAVE_CUDA), true)
  $(warning %%% WARNING!!!)
  $(warning %%% CUDA not found! Incorrect path in CUDA_TK_BASE: $(CUDA_TK_BASE))
  $(warning %%% Try setting CUDA_TK_BASE in 'trunk/src/tnet.mk')
  $(warning %%% WARNING!!!)
else
  #$(warning %%% INFO: Using CUDA from CUDA_TK_BASE: $(CUDA_TK_BASE))
endif


##### Includes
INCLUDE := -IKaldiLib -ITNetLib -ISTKLib 
INCLUDE += -ICuBaseLib -ICuTNetLib
INCLUDE += -I$(CUDA_TK_BASE)/include

CXXFLAGS += $(INCLUDE)

##### CPU implementation libs
LDFLAGS :=   -LTNetLib -lTNetLib
LDFLAGS +=   -LKaldiLib -lKaldiLib
LDFLAGS +=   -pthread 


##### Link one of the BLASes
ifeq ($(BLAS), ATLAS)
  #Link statically with pre-compiled ATLAS
  ifeq ($(BITS64), true)
    ATLAS_DIR=$(PWD)/ATLAS/64/
    BLAS_LDFLAGS = $(ATLAS_DIR)/libclapack.a $(ATLAS_DIR)/libcblas.a $(ATLAS_DIR)/libatlas.a $(ATLAS_DIR)/libf77blas.a
  else
    ATLAS_DIR=$(PWD)/ATLAS/32/
    BLAS_LDFLAGS = $(ATLAS_DIR)/libclapack.a $(ATLAS_DIR)/libcblas.a $(ATLAS_DIR)/libatlas.a $(ATLAS_DIR)/libf77blas.a
  endif
endif
ifeq ($(BLAS), GotoBLAS)
  #Link dynamically with pre-compiled GotoBLAS (It had threading issue at ICSI)
  ifeq ($(BITS64), true) 
    BLAS_LDFLAGS = -LGotoBLASLib -lgoto2_64 -lgfortran -Wl,-rpath,$(PWD)/GotoBLASLib
  else
    BLAS_LDFLAGS = -LGotoBLASLib -lgoto2 -lgfortran -Wl,-rpath,$(PWD)/GotoBLASLib
  endif
endif
ifeq ($(BLAS), MKL) 
  #Link statically with MKL
  ifeq ($(BITS64), true) 
    BLAS_LDFLAGS = $(MKLROOT)/lib/em64t/libmkl_solver_lp64_sequential.a -Wl,--start-group  $(MKLROOT)/lib/em64t/libmkl_intel_lp64.a $(MKLROOT)/lib/em64t/libmkl_sequential.a $(MKLROOT)/lib/em64t/libmkl_core.a -Wl,--end-group -lpthread -lm
  else
    BLAS_LDFLAGS = $(MKLROOT)/lib/32/libmkl_solver_sequential.a -Wl,--start-group  $(MKLROOT)/lib/32/libmkl_intel.a $(MKLROOT)/lib/32/libmkl_sequential.a $(MKLROOT)/lib/32/libmkl_core.a -Wl,--end-group -lpthread -lm 
  endif
endif

ifndef BLAS_LDFLAGS
  $(error %%% Unsupported BLAS selected \"$(BLAS)\" in tnet.mk, options:ATLAS,GotoBLAS,MKL)
endif

#Add BLAS to linker options
LDFLAGS += $(BLAS_LDFLAGS)



##### CUDA implementation libs
ifeq ($(CUDA), true)
  #TNet libs
  LDFLAGS_CUDA := -LCuTNetLib -lCuTNet
  LDFLAGS_CUDA += -LCuBaseLib -lCuBase
  #CUDA toolkit libs
  ifeq ($(BITS64), true)
    LDFLAGS_CUDA += -L$(CUDA_TK_BASE)/lib64 -Wl,-rpath,$(CUDA_TK_BASE)/lib64
  else
    LDFLAGS_CUDA += -L$(CUDA_TK_BASE)/lib -Wl,-rpath,$(CUDA_TK_BASE)/lib
  endif
  LDFLAGS_CUDA += -lcublas -lcudart -lcuda 
endif


##############################################################
# Target programs 
##############################################################

#CPU tools
BINS := TNet TNorm TFeaCat TSegmenter TJoiner
all : $(BINS) 
$(BINS): lib

#GPU tools
CUBINS := TNetCu TNormCu TFeaCatCu TRbmCu
ifeq ($(STK), true)
  CUBINS += TMpeCu TMmiCu
endif
ifeq ($(CUDA), true)
cubins : $(CUBINS)
##HINT: Link CUDA libs only with tools using CUDA!!!##
##(recursive target-specific variable value)##
cubins : LDFLAGS += $(LDFLAGS_CUDA)
##
all : cubins
$(CUBINS): lib culib
endif


##############################################################
# program compliling implicit rule
##############################################################
% : %.o
	$(CXX)  -o $@  $< $(CXXFLAGS) $(INCLUDE) $(LDFLAGS)

 
##############################################################
# module compliling implicit rule
##############################################################
%.o : %.cc lib
	$(CXX)  -o $@  -c $< $(CFLAGS) $(CXXFLAGS) $(INCLUDE)


##############################################################
# STK specific rules
##############################################################
#TMpeCu depends on STK
TMpeCu.o: stklib
TMpeCu: LDFLAGS := -LSTKLib -lSTKLib $(LDFLAGS) $(LDFLAGS_CUDA)
#TMmiCu depends on STK
TMmiCu.o: stklib
TMmiCu: LDFLAGS := -LSTKLib -lSTKLib $(LDFLAGS) $(LDFLAGS_CUDA)


##############################################################
# Source files for CPU/GPU tools
##############################################################
CC_BINS=$(addsuffix .cc, $(BINS))
CC_CUBINS=$(addsuffix .cc, $(CUBINS))

O_BINS=$(addsuffix .o, $(BINS))
O_CUBINS=$(addsuffix .o, $(CUBINS))

$(O_BINS) : $(CC_BINS) 
$(O_CUBINS) : $(CC_CUBINS) 

$(BINS) : $(O_BINS)
$(CUBINS) : $(O_CUBINS)

##############################################################
.PHONY: lib culib stklib clean doc depend

lib:
	@cd KaldiLib && make $(FWDPARAM)
	@cd TNetLib && make $(FWDPARAM)

culib: 
	@cd CuBaseLib && make $(FWDPARAM)
	@cd CuTNetLib && make $(FWDPARAM)
	
stklib:
	@cd STKLib && make $(FWDPARAM)

clean:
	rm -f *.o $(BINS) $(CUBINS)
	@cd STKLib && make clean
	@cd KaldiLib && make clean
	@cd TNetLib && make clean
	@cd CuBaseLib && make clean
	@cd CuTNetLib && make clean

doc:
	doxygen ../doc/doxyfile_TNet

depend: 
	$(CXX) -M $(CXXFLAGS) $(CC_BINS) $(INCLUDE) > .depend.mk1
	@cd KaldiLib && make depend
	@cd TNetLib && make depend
	touch .depend.mk{1,2}
	cat .depend.mk{1,2} > .depend.mk
	rm .depend.mk{1,2}

cudepend:
	$(CXX) -M $(CXXFLAGS) $(CC_CUBINS) $(INCLUDE) > .depend.mk2
	@cd CuBaseLib && make depend
	@cd CuTNetLib && make depend
ifeq ($(HAVE_CUDA), true)
depend: cudepend
endif
ifeq ($(STK), true)
cudepend: stklib
endif


-include .depend.mk


