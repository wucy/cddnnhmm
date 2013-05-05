
#include "cuActivation.h"
#include "cumath.h"


namespace TNet {


  void
  CuSigmoid::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    CuMath<BaseFloat>::Sigmoid(Y, X);
  }


  void 
  CuSigmoid::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    CuMath<BaseFloat>::DiffSigmoid(Y, X, mOutput);
  }



  void 
  CuSoftmax::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    CuMath<BaseFloat>::Softmax(Y,X);
  }

   
   
  void
  CuSoftmax::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    //we assume X is already dE/dSoftmax_input
    Y.CopyFrom(X);
  }



  //// BlockSoftmax
  void CuBlockSoftmax::ReadFromStream(std::istream& rIn) {
    rIn >> mDim; 
    mDimOffset.Init(mDim.Dim()+1);
    
    int off=0; 
    for(int i=0; i<mDim.Dim(); i++) { 
      mDimOffset[i]=off;
      off += mDim[i];
    }
    mDimOffset[mDim.Dim()]=off;

    if(off!=GetNOutputs()) {
      KALDI_ERR << "Non-matching dimension of sum of softmaxes,"
        << " the sum:" << off 
        << " GetNOutputs:" << GetNOutputs();
    }
  }

  void CuBlockSoftmax::WriteToStream(std::ostream& rOut) {
    rOut << mDim;
  }

  void CuBlockSoftmax::PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y) {
    CuMatrix<BaseFloat> X_stripe;
    CuMatrix<BaseFloat> Y_stripe;
    
    for(int j=0; j<mDim.Dim(); j++) {
      X_stripe.Init(X.Rows(),mDim[j]);
      Y_stripe.Init(Y.Rows(),mDim[j]);
      X_stripe.CopyCols(mDim[j],mDimOffset[j],X,0);
      //process block of softmax
      CuMath<BaseFloat>::Softmax(Y_stripe,X_stripe);
      //copy the result back
      Y.CopyCols(mDim[j],0,Y_stripe,mDimOffset[j]);
    }
  }


  void CuBlockSoftmax::BackpropagateFnc(const CuMatrix<BaseFloat>& X_gpu, CuMatrix<BaseFloat>& Y_gpu) {
    //
    //DO THE BACKPROPAGATE STEP ON CPU
    //it is hard to do it in GPU!
    //
    Matrix<BaseFloat> X(X_gpu.Rows(),X_gpu.Cols());
    Matrix<BaseFloat> Y(Y_gpu.Rows(),Y_gpu.Cols());
    //copy from devide to host
    X_gpu.CopyTo(X);
    //set the output to zero
    Y.Zero();

    //we will measure the time so we know how slow it is
    Timer tim; tim.Start();

    //copy only parts of the error
    //from softmax intervals which sum up to 0.0, not 1.0
    for(int i=0; i<X.Rows(); i++) {
      for(int j=0; j<mDim.Dim(); j++) {
        const BfSubVector x_i_smx_j(X[i].Range(mDimOffset[j],mDim[j]));
        BaseFloat sum = x_i_smx_j.Sum();
        if(sum > -0.1 && sum < 0.1) {
          BfSubVector y_i_smx_j(Y[i].Range(mDimOffset[j],mDim[j]));
          y_i_smx_j.Copy(x_i_smx_j);
        } else if (sum > 0.9 && sum < 1.1) {
          ; //do nothing
        } else {
          KALDI_ERR << "Invalid sum: " << sum;
        }
      }
    }

    //we are done, end the timer
    tim.End(); CuDevice::Instantiate().AccuProfile("BlockSoftmax::BackpropagateFnc",tim.Val());

    //finally copy the output to the GPU
    Y_gpu.CopyFrom(Y);

  }



} //namespace

