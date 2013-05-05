

#include "cuSharedLinearity.h"
#include "cumath.h"


namespace TNet
{

  void 
  CuSharedLinearity::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    CuMath<BaseFloat>::VecExpand(mBias,mBiasExpand); /// [ 1 2 3 ] -> [ 1 2 3 1 2 3 ... ]
    Y.AddScaledRow(1.0,mBiasExpand,0.0);

    //mBiasExpand.Print();

    for(int i=0; i<mNInstances; i++) {
      CuMath<BaseFloat>::OffsetGemm('N','N', 1.0, X, mLinearity, 1.0, Y, 
                                    i*mLinearity.Rows(), 0, i*mLinearity.Cols());
    }
    //KALDI_COUT << CuDevice::Instantiate().GetFreeMemory();
    //GetInput().Print();
    //GetOutput().Print();
  }


  void 
  CuSharedLinearity::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    for(int i=0; i<mNInstances; i++) {
      CuMath<BaseFloat>::OffsetGemm('N', 'T', 1.0, X, mLinearity, 0.0, Y,
                                    i*mLinearity.Cols(), 0, i*mLinearity.Rows());
    }
  }

  
  void 
  CuSharedLinearity::
  Update() 
  {
    //gain from the momentum
    BaseFloat mmt_gain = static_cast<BaseFloat>(1.0/(1.0-mMomentum));

    //get gradient of shared linearity
    for(int i=0; i<mNInstances; i++) {
      CuMath<BaseFloat>::OffsetGemm('T','N',1.0,
                        GetInput(),GetErrorInput(),
                        ((i==0)?mMomentum:1.0f), mLinearityCorrection, 
                        i*mLinearity.Rows(),i*mLinearity.Cols(),0);
    }
    //get gradient of shared bias
    mBiasCorrectionExpand.AddColSum(1.0,GetErrorInput(),0.0);
    CuMath<BaseFloat>::VecAddColSum(1.0,mBiasCorrectionExpand,mMomentum,mBiasCorrection);
   
    //perform update 
    //(divide learning rate by number of instances)
    mLinearity.AddScaled(-mLearningRate/mmt_gain/mNInstances,mLinearityCorrection,1.0);
    mBias.AddScaled(-mLearningRate/mmt_gain/mNInstances,mBiasCorrection,0.0);
    
    //regularize by weight decay
    BaseFloat L2_decay = -mLearningRate*mWeightcost*GetInput().Rows();
    mLinearity.AddScaled(L2_decay,mLinearity,1.0);
  }


  void
  CuSharedLinearity::
  ReadFromStream(std::istream& rIn)
  {
    //number of instances of shared weights in layer
    rIn >> std::ws >> mNInstances;
    if(mNInstances < 1) {
      KALDI_ERR << "Bad number of instances:" << mNInstances;
    }
    if(GetNInputs() % mNInstances != 0 || GetNOutputs() % mNInstances != 0) {
      KALDI_ERR << "Number of Inputs/Outputs must be divisible by number of instances"
                << " Inputs:" << GetNInputs()
                << " Outputs" << GetNOutputs()
                << " Intances:" << mNInstances;
    }
      
    //matrix is stored transposed as SNet does
    BfMatrix transpose;
    rIn >> transpose;
    mLinearity.CopyFrom(BfMatrix(transpose, TRANS));
    //biases stored normally
    BfVector bias;
    rIn >> bias;
    mBias.CopyFrom(bias);

    if(transpose.Cols()*transpose.Rows() == 0) {
      KALDI_ERR << "Missing linearity matrix in network file";
    }
    if(bias.Dim() == 0) {
      KALDI_ERR << "Missing bias vector in network file";
    }


    if(mLinearity.Cols() != GetNOutputs() / mNInstances || 
       mLinearity.Rows() != GetNInputs() / mNInstances ||
       mBias.Dim() != GetNOutputs() / mNInstances
    ){
      KALDI_ERR << "Wrong dimensionalities of matrix/vector in network file\n"
                << "Inputs:" << GetNInputs()
                << "Outputs:" << GetNOutputs()
                << "\n"
                << "linearityCols:" << mLinearity.Cols()
                << "linearityRows:" << mLinearity.Rows()
                << "biasDims:" << mBias.Dim()
                << "\n";
    }

    mLinearityCorrection.Init(mLinearity.Rows(),mLinearity.Cols());
    mBiasCorrection.Init(mBias.Dim());

    mBiasExpand.Init(mBias.Dim()*mNInstances);
    mBiasCorrectionExpand.Init(mBias.Dim()*mNInstances);
  }

   
  void
  CuSharedLinearity::
  WriteToStream(std::ostream& rOut)
  {
    rOut << mNInstances << std::endl;

    //matrix is stored transposed as SNet does
    BfMatrix tmp;
    mLinearity.CopyTo(tmp);
    BfMatrix transpose(tmp, TRANS);
    rOut << transpose;
    //biases stored normally
    BfVector vec;
    mBias.CopyTo(vec);
    rOut << vec;
    rOut << std::endl;
  }

 
} //namespace
