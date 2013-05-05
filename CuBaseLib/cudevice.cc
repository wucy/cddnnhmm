
#include <cudevice.h>
#include <cublas.h>
#include <cuda.h>

///////////////////
//DEBUG: Just make sure it compiles...
#include "cumatrix.h"
#include "cuvector.h"
#include "cumath.h"
template class TNet::CuMatrix<float>;
template class TNet::CuVector<float>;
template class TNet::CuMath<float>;
///////////////////

namespace TNet {


  /**********************************************************************************
   * CuDevice::
   */
  CuDevice::
  CuDevice()
    : mIsPresent(false), mVerbose(false)
  {

    //get number of devices
    int N_GPU = 0;
    cudaGetDeviceCount(&N_GPU);

    //check whether the system use compute exclusive mode
    //free GPU gets selected automatically)
    bool do_selection = true;
    if(N_GPU > 0) {
      cudaError_t e;
      int dev;
      e = cudaThreadSynchronize();//create context
      if(e != cudaSuccess) {
        KALDI_ERR << "Failed to create CUDA context on a GPU";
      }
      //get the gpu_id and device properties
      e = cudaGetDevice(&dev);
      if(e != cudaSuccess) {
        KALDI_ERR << "Failed to get current device";
      }
      struct cudaDeviceProp dev_prop;
      e = cudaGetDeviceProperties(&dev_prop,dev);
      if(e != cudaSuccess) {
        KALDI_ERR << "Failed to get device properties";
      }
      //check for the compute exclusive mode
      switch(dev_prop.computeMode) {
        case cudaComputeModeExclusive:
        #if (CUDA_VERSION >= 4000)
        case cudaComputeModeExclusiveProcess :
        #endif
        {
          KALDI_COUT << "Compute exclusive mode detected. "
                    << "The free GPU will be automatically selected by OS/Driver Will be using device(" << dev << "): "
                    << dev_prop.name << "\n";
          do_selection = false;//and keep the context
        }
        break;
        default:
          //we will use selection by free memory
          cudaThreadExit();//destroy context
      }
    }

    //select device if more than one
    if(N_GPU > 1 && do_selection) {
      char name[128];
      #if (CUDA_VERSION >= 3020)
      size_t free, total;
      #else
      unsigned int free, total;
      #endif
      std::vector<float> free_mem_ratio;
      //get ratios of memory use
      KALDI_COUT << "Selecting from " << N_GPU << " GPUs\n";
      for(int n=0; n<N_GPU; n++) {
        KALDI_COUT << "cudaSetDevice(" << n << "): ";
        switch(cudaSetDevice(n)) {
          case cudaSuccess :
          {
            cudaThreadSynchronize();//create context
            cuDeviceGetName(name,128,n);
            KALDI_COUT << name << "\t";
            cuSafeCall(cuMemGetInfo(&free,&total));
            KALDI_COUT << "free: " << free/1024/1024 << "M, "
                      << "total: "<< total/1024/1024 << "M, "
                      << "ratio: "<< free/(float)total << "\n";
            free_mem_ratio.push_back(free/(float)total);
            cudaThreadExit();//destroy context
          } break;
          default:
          { 
            KALDI_COUT << "failed...";
            free_mem_ratio.push_back(0.0);
          }
        }
      }
      //find GPU with max free memory
      int max_id=0;
      for(int n=1; n<free_mem_ratio.size(); n++) {
        if(free_mem_ratio[n] > free_mem_ratio[max_id]) max_id=n;
      }
      KALDI_COUT << "Selected device: " << max_id << " (automatically)\n";
      cuSafeCall(cudaSetDevice(max_id));
    }
      
    if(N_GPU > 0) {
      //initialize the CUBLAS
      cuSafeCall(cublasInit());
      mIsPresent = true;
    } else {
      KALDI_WARN << "No CUDA enabled GPU is present!";
    }
  }

  CuDevice::
  ~CuDevice()
  {
    if(mIsPresent) {
      cuSafeCall(cublasShutdown());
      if(mVerbose) {
        KALDI_LOG << "CUBLAS released";
        PrintProfile();
      }
    } else {
      KALDI_WARN << "No CUDA enabled GPU was present!";
    }
  }


  void 
  CuDevice::
  SelectGPU(int gpu_id)
  {
    //get number of devices
    int N_GPU = 0;
    cudaGetDeviceCount(&N_GPU);
    if(gpu_id >= N_GPU) {
      KALDI_ERR << "Cannot select GPU " << gpu_id 
                << ", detected " << N_GPU << " CUDA capable cards!";
    }
    //release old card
    cuSafeCall(cublasShutdown());
    cudaThreadExit();
    //select new card
    cuSafeCall(cudaSetDevice(gpu_id));
    //initialize CUBLAS
    cuSafeCall(cublasInit());
    KALDI_COUT << "Selected device " << gpu_id << " (manually)\n";
  }


  std::string
  CuDevice::
  GetFreeMemory()
  {
    #if (CUDA_VERSION >= 3020)
    size_t free, total;
    #else
    unsigned int free, total;
    #endif
    cuMemGetInfo(&free, &total);
    std::ostringstream os;
    os << "Free:" << free/(1024*1024) << "MB "
       << "Used:" << (total-free)/(1024*1024) << "MB "
       << "Total:" << total/(1024*1024) << "MB";
    return os.str();
  }


  ////////////////////////////////////////////////
  // Instance of the static singleton 
  //
  CuDevice CuDevice::msDevice;
  //
  ////////////////////////////////////////////////
  


}


