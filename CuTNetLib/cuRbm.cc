
#include <string>
#include <sstream>

#include "cuRbm.h"

#include "cumath.h"


namespace TNet
{

  void 
  CuRbm::
  PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    Y.SetConst(0.0);
    Y.AddScaledRow(1.0,mHidBias,0.0);
    Y.Gemm('N','N', 1.0, X, mVisHid, 1.0);
    if(mHidType == BERNOULLI) {
      CuMath<BaseFloat>::Sigmoid(Y,Y);
    }
  }


  void 
  CuRbm::
  BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
  {
    KALDI_ERR << "Please, convert the RBM to <biasedlinearity> and <sigmoid>,"
                 "before training by SGD";
  }

  
  void 
  CuRbm::
  Update() 
  {
    KALDI_ERR << "Please, convert the RBM to <biasedlinearity> and <sigmoid>,"
                 "before training by SGD";
  }


  void 
  CuRbm::
  Propagate(const CuMatrix<BaseFloat>& visProbs, CuMatrix<BaseFloat>& hidProbs)
  {
    if(visProbs.Cols() != GetNInputs()) {
      KALDI_ERR << " Nonmatching input dim, needs:" << GetNInputs() 
                << " got:" << visProbs.Cols() << "\n";
    }

    hidProbs.Init(visProbs.Rows(),GetNOutputs());

    PropagateFnc(visProbs, hidProbs);
  }


  void
  CuRbm::
  Reconstruct(const CuMatrix<BaseFloat>& hidState, CuMatrix<BaseFloat>& visProbs)
  {
    visProbs.Init(hidState.Rows(),mNInputs);
    visProbs.SetConst(0.0);
    visProbs.AddScaledRow(1.0,mVisBias,0.0);
    visProbs.Gemm('N','T', 1.0, hidState, mVisHid, 1.0);
    if(mVisType == BERNOULLI) {
      CuMath<BaseFloat>::Sigmoid(visProbs,visProbs);
    }
  }


  void 
  CuRbm::
  RbmUpdate(const CuMatrix<BaseFloat>& pos_vis, const CuMatrix<BaseFloat>& pos_hid, const CuMatrix<BaseFloat>& neg_vis, const CuMatrix<BaseFloat>& neg_hid)
  {
    assert(pos_vis.Rows() == pos_hid.Rows() &&
           pos_vis.Rows() == neg_vis.Rows() &&
           pos_vis.Rows() == neg_hid.Rows() &&
           pos_vis.Cols() == neg_vis.Cols() &&
           pos_hid.Cols() == neg_hid.Cols() &&
           pos_vis.Cols() == mNInputs &&
           pos_hid.Cols() == mNOutputs);

    //  UPDATE vishid matrix
    //  
    //  vishidinc = momentum*vishidinc + ...
    //              epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    //
    //  vishidinc[t] = -(epsilonw/numcases)*negprods + momentum*vishidinc[t-1]
    //                 +(epsilonw/numcases)*posprods
    //                 -(epsilonw*weightcost)*vishid[t-1]
    //
    BaseFloat N = static_cast<BaseFloat>(pos_vis.Rows());
    mVisHidCorrection.Gemm('T','N',-mLearningRate/N,neg_vis,neg_hid,mMomentum);
    mVisHidCorrection.Gemm('T','N',+mLearningRate/N,pos_vis,pos_hid,1.0);
    mVisHidCorrection.AddScaled(-mLearningRate*mWeightcost,mVisHid,1.0);
    mVisHid.AddScaled(1.0,mVisHidCorrection,1.0);

    //  UPDATE visbias vector
    //
    //  visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    //
    mVisBiasCorrection.AddColSum(-mLearningRate/N,neg_vis,mMomentum);
    mVisBiasCorrection.AddColSum(+mLearningRate/N,pos_vis,1.0);
    mVisBias.AddScaled(1.0,mVisBiasCorrection,1.0);
    
    //  UPDATE hidbias vector
    //
    // hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    //
    mHidBiasCorrection.AddColSum(-mLearningRate/N,neg_hid,mMomentum);
    mHidBiasCorrection.AddColSum(+mLearningRate/N,pos_hid,1.0);
    mHidBias.AddScaled(1.0/*0.0*/,mHidBiasCorrection,1.0);

  }


  void
  CuRbm::
  ReadFromStream(std::istream& rIn)
  {
    //type of the units
    std::string str;
    
    rIn >> std::ws >> str;
    if(0 == str.compare("bern")) {
      mVisType = BERNOULLI;
    } else if(0 == str.compare("gauss")) {
      mVisType = GAUSSIAN;
    } else KALDI_ERR << "Invalid unit type: " << str;

    rIn >> std::ws >> str;
    if(0 == str.compare("bern")) {
      mHidType = BERNOULLI;
    } else if(0 == str.compare("gauss")) {
      mHidType = GAUSSIAN;
    } else KALDI_ERR << "Invalid unit type: " << str;


    //matrix is stored transposed as SNet does
    BfMatrix transpose;
    rIn >> transpose;
    mVisHid.CopyFrom(BfMatrix(transpose, TRANS));
    //biases stored normally
    BfVector bias;
    rIn >> bias;
    mVisBias.CopyFrom(bias);
    rIn >> bias;
    mHidBias.CopyFrom(bias);
  }

   
  void
  CuRbm::
  WriteToStream(std::ostream& rOut)
  {
    //store unit type info
    if(mVisType == BERNOULLI) {
      rOut << " bern ";
    } else {
      rOut << " gauss ";
    }
    if(mHidType == BERNOULLI) {
      rOut << " bern\n";
    } else {
      rOut << " gauss\n";
    }

    //matrix is stored transposed as SNet does
    BfMatrix tmp;
    mVisHid.CopyTo(tmp);
    BfMatrix transpose(tmp, TRANS);
    rOut << transpose;
    //biases stored normally
    BfVector vec;
    mVisBias.CopyTo(vec);
    rOut << vec;
    rOut << std::endl;
    mHidBias.CopyTo(vec);
    rOut << vec;
    rOut << std::endl;
  }

 
} //namespace
