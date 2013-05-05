

#include "cuBlockdiagonalLinearity.h"
#include "cumath.h"

namespace TNet
{

  void 
  CuBlockdiagonalLinearity::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    //Y.SetConst(0.0);

    //precopy bias
    Y.AddScaledRow(1.0,mBias,0.0);

    //mulitply with the matrices
    int offset_in=0, offset_out=0;
    for (int i=0; i<mNBlocks; i++) {
      CuMath<BaseFloat>::OffsetGemm('N','N', 1.0, X, mLinearity[i], 1.0, Y, 
                                    offset_in, 0, offset_out);
      offset_in += mLinearity[i].Rows();
      offset_out += mLinearity[i].Cols();
    }
  }


  void 
  CuBlockdiagonalLinearity::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    //Y.SetConst(0.0);

    int offset_in=0, offset_out=0;
    for(int i=0; i<mNBlocks; i++) {
      CuMath<BaseFloat>::OffsetGemm('N', 'T', 1.0, X, mLinearity[i], 0.0, Y,
                                    offset_in, 0, offset_out);
      offset_in += mLinearity[i].Cols();
      offset_out += mLinearity[i].Rows();
    }
  }

  
  void 
  CuBlockdiagonalLinearity::
  Update() 
  {
    //new implementation
    BaseFloat mmt_gain = static_cast<BaseFloat>(1.0/(1.0-mMomentum));

    //get gradients of discrete linearities
    int offset_in=0, offset_out=0;
    for(int i=0; i<mNBlocks; i++) {
      CuMath<BaseFloat>::OffsetGemm('T','N',1.0,
                        GetInput(),GetErrorInput(),
                        mMomentum, mLinearityCorrection[i],
                        offset_in,offset_out,0);
      offset_in += mLinearity[i].Rows();
      offset_out += mLinearity[i].Cols();
    }
    for(int i=0; i<mNBlocks; i++) {
      //perform update 
      mLinearity[i].AddScaled(-mLearningRate/mmt_gain,mLinearityCorrection[i],1.0);
      //regularize by weight decay
      BaseFloat L2_decay = -mLearningRate*mWeightcost*GetInput().Rows();
      mLinearity[i].AddScaled(L2_decay,mLinearity[i],1.0);
    }

    //get gradient of bias
    mBiasCorrection.AddColSum(1.0,GetErrorInput(),mMomentum);
    //update biases
    mBias.AddScaled(-mLearningRate/mmt_gain,mBiasCorrection,1.0);
  }


  void
  CuBlockdiagonalLinearity::
  ReadFromStream(std::istream& rIn)
  {
    rIn >> std::ws >> mNBlocks;
    if(mNBlocks < 1) {
      KALDI_ERR << "Bad number of blocks:" << mNBlocks;
    }

    mLinearity.resize(mNBlocks);
    mLinearityCorrection.resize(mNBlocks);

    int in_dim = 0, out_dim = 0;
    for(int i=0; i<mNBlocks; i++) {
      //matrix is stored transposed as SNet does
      BfMatrix transpose;
      rIn >> transpose;
      mLinearity[i].CopyFrom(BfMatrix(transpose, TRANS));
      
      if(transpose.Cols()*transpose.Rows() == 0) {
        KALDI_ERR << "Missing linearity matrix in network file";
      }
      //allocate training buffers
      mLinearityCorrection[i].Init(mLinearity[i].Rows(),mLinearity[i].Cols());
      mLinearityCorrection[i].SetConst(0.0);

      in_dim += transpose.Cols();
      out_dim += transpose.Rows();
    }
    
    //biases stored normally
    BfVector bias;
    rIn >> bias;
    mBias.CopyFrom(bias);
    if(bias.Dim() == 0) {
      KALDI_ERR << "Missing bias vector in network file";
    }
    mBiasCorrection.Init(mBias.Dim());
    mBiasCorrection.SetConst(0.0);

    if(out_dim != GetNOutputs() || 
       in_dim != GetNInputs() ||
       mBias.Dim() != GetNOutputs()
    ){
      KALDI_ERR << "Wrong dimensionalities of matrix/vector in network file\n"
                << "Inputs:" << GetNInputs()
                << "Outputs:" << GetNOutputs()
                << "\n"
                << "linearityCols:" << in_dim
                << "linearityRows:" << out_dim
                << "biasDims:" << mBias.Dim()
                << "\n";
    }
  }

   
  void
  CuBlockdiagonalLinearity::
  WriteToStream(std::ostream& rOut)
  {
    rOut << mNBlocks << "\n";
    for(int i=0; i< mNBlocks; i++) {
      //matrix is stored transposed as SNet does
      BfMatrix tmp;
      mLinearity[i].CopyTo(tmp);
      BfMatrix transpose(tmp, TRANS);
      rOut << transpose;
    }
    //biases stored normally
    BfVector vec;
    mBias.CopyTo(vec);
    rOut << vec;
    rOut << std::endl;
  }

 
} //namespace

