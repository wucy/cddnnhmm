

#include "cuBiasedLinearity.h"


namespace TNet
{

  void 
  CuBiasedLinearity::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    //Y.SetConst(0.0);
    Y.AddScaledRow(1.0,mBias,0.0);
    Y.Gemm('N','N', 1.0, X, mLinearity, 1.0);
  }


  void 
  CuBiasedLinearity::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    //Y.SetConst(0.0);
    Y.Gemm('N', 'T', 1.0, X, mLinearity, 0.0);
  }

  
  void 
  CuBiasedLinearity::
  Update() 
  {
    //we will compensate for the learning rate augmented by momentum
    BaseFloat mmt_gain = static_cast<BaseFloat>(1.0/(1.0-mMomentum));

    //get the gradients
    mLinearityCorrection.Gemm('T','N',1.0,GetInput(),GetErrorInput(),mMomentum);
    mBiasCorrection.AddColSum(1.0,GetErrorInput(),mMomentum);

    //update with the gradients
    mLinearity.AddScaled(-mLearningRate/mmt_gain,mLinearityCorrection,1.0);
    mBias.AddScaled(-mLearningRate/mmt_gain,mBiasCorrection,1.0);

    //regularize by weight decay (from actual weights only)
    BaseFloat L2_decay = -mLearningRate*mWeightcost*GetInput().Rows();
    mLinearity.AddScaled(L2_decay, mLinearity,1.0);
  }


  void
  CuBiasedLinearity::
  ReadFromStream(std::istream& rIn)
  {
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
    if(mLinearity.Cols() != GetNOutputs() || 
       mLinearity.Rows() != GetNInputs() ||
       mBias.Dim() != GetNOutputs()
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
  }

   
  void
  CuBiasedLinearity::
  WriteToStream(std::ostream& rOut)
  {
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

