
#include <algorithm>
//#include <locale>
#include <cctype>
#include <list>
#include <sstream>

#include "cuNnet.h"

#include "cuBlockdiagonalLinearity.h"
#include "cuSharedLinearity.h"
#include "cuSparseLinearity.h"
#include "cuRbm.h"
#include "cuRbmSparse.h"
#include "cuBlockArray.h"

namespace TNet {


  void CuNetwork::Feedforward(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>& out, 
                          size_t start_frm_ext, size_t end_frm_ext) {
    //empty network: copy input to output 
    if(mNnet.size() == 0) {
      if(out.Rows() != in.Rows() || out.Cols() != in.Cols()) {
        out.Init(in.Rows(),in.Cols());
      }
      out.CopyFrom(in);
      return;
    }
    
    //short input: propagate in one block  
    if(in.Rows() < 5000) { 
      Propagate(in,out);
    } else {//long input: propagate per parts
      //initialize
      out.Init(in.Rows(),GetNOutputs());
      CuMatrix<BaseFloat> tmp_in, tmp_out;
      int done=0, block=1024;
      //propagate first part
      tmp_in.Init(block+end_frm_ext,in.Cols());
      // tmp_in.Copy(in.Range(0,block+end_frm_ext,0,in.Cols()));
      tmp_in.CopyRows(block+end_frm_ext, 0, in, 0);
      Propagate(tmp_in,tmp_out);
      // out.Range(0,block,0,tmp_out.Cols()).Copy(
      //  tmp_out.Range(0,block,0,tmp_out.Cols())
      //);
      out.CopyRows(block,0,tmp_out,0);
      done += block;
      //propagate middle parts
      while((done+2*block) < in.Rows()) {
        tmp_in.Init(block+start_frm_ext+end_frm_ext,in.Cols());
        //tmp_in.Copy(in.Range(done-start_frm_ext, block+start_frm_ext+end_frm_ext, 0,in.Cols()));
        tmp_in.CopyRows(block+start_frm_ext+end_frm_ext, done-start_frm_ext, in, 0);
        Propagate(tmp_in,tmp_out);
        //out.Range(done,block,0,tmp_out.Cols()).Copy(
        //  tmp_out.Range(start_frm_ext,block,0,tmp_out.Cols())
        //);
        out.CopyRows(block, start_frm_ext, tmp_out, done);
        done += block;
      }
      //propagate last part
      tmp_in.Init(in.Rows()-done+start_frm_ext,in.Cols());
      //tmp_in.Copy(in.Range(done-start_frm_ext,in.Rows()-done+start_frm_ext,0,in.Cols()));
      tmp_in.CopyRows(in.Rows()-done+start_frm_ext,done-start_frm_ext,in,0);
      Propagate(tmp_in,tmp_out);
      //out.Range(done,out.Rows()-done,0,out.Cols()).Copy(
      //  tmp_out.Range(start_frm_ext,tmp_out.Rows()-start_frm_ext,0,tmp_out.Cols())   
      //);
      out.CopyRows(out.Rows()-done, start_frm_ext, tmp_out, done);

      done += tmp_out.Rows()-start_frm_ext;
      assert(done == out.Rows());
    }
  }



  void
  CuNetwork::
  ReadNetwork(const char* pSrc)
  {
    std::ifstream in(pSrc);
    if(!in.good()) {
      KALDI_ERR << "Cannot read model: " << pSrc;
    }
    ReadNetwork(in);
    in.close();
  }

 
 
  void
  CuNetwork::
  WriteNetwork(const char* pDst)
  {
    std::ofstream out(pDst);
    if(!out.good()) {
      KALDI_ERR << "Cannot write model: " << pDst;
    }
    WriteNetwork(out);
    out.close();
  }

   

  void
  CuNetwork::
  ReadNetwork(std::istream& rIn)
  {
    //get the network elements from a factory
    CuComponent *pComp;
    while(NULL != (pComp = ComponentFactory(rIn))) { 
      mNnet.push_back(pComp);
    }
  }



  void
  CuNetwork::
  WriteNetwork(std::ostream& rOut)
  {
    //dump all the componetns
    LayeredType::iterator it;
    for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
      ComponentDumper(rOut, **it);
    }
  }


  void
  CuNetwork::
  SetLearnRate(BaseFloat learnRate, const char* pLearnRateFactors)
  {
    //parse the learn rate factors: "0.1:0.5:0.6:1.0" to std::list
    std::list<BaseFloat> lr_factors;

    if(NULL != pLearnRateFactors) {
      //replace any separator by ' '
      std::string str(pLearnRateFactors);
      size_t pos = 0;
      while((pos = str.find_first_not_of("0123456789.+-eE ")) != std::string::npos)
        str[pos] = ' ';

      //parse to std::list
      std::istringstream is(str);
      is >> std::skipws;
      BaseFloat f; 
      while(!is.eof()) {
        if(!(is >> f).fail()) { lr_factors.push_back(f); }
        else break;
      }

      //count updatable components
      size_t updatable_components = 0;
      for(int i=0; i<mNnet.size(); i++) {
        if(mNnet[i]->IsUpdatable()) updatable_components++;
      }
      //make sure the count is same as number of factors
      if(updatable_components != lr_factors.size()) {
        KALDI_ERR << "Wrong number of lr_factors : " << pLearnRateFactors
                  << " the string contains " << lr_factors.size() << " factors"
                  << " while the MLP contains " << updatable_components 
                  << " updatable componets!";
      }
    }


    //give scaled learning rate to components
    BaseFloat lr_factor;
    LayeredType::iterator it;
    for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
      if((*it)->IsUpdatable()) {
        //get next scale factor
        if(NULL != pLearnRateFactors) {
          lr_factor = lr_factors.front(); 
          lr_factors.pop_front(); 
        } else {
          lr_factor = 1.0;
        }
        //set scaled learning rate to the component
        dynamic_cast<CuUpdatableComponent*>(*it)->LearnRate(learnRate*lr_factor);
      }
    }

    //store global learning rate
    mGlobLearnRate = learnRate;
    mpLearnRateFactors = pLearnRateFactors;
  }


  BaseFloat
  CuNetwork::
  GetLearnRate()
  {
    return mGlobLearnRate;
  }


  void
  CuNetwork::
  PrintLearnRate()
  {
    assert(mNnet.size() > 0);
    KALDI_COUT << "Learning rate: global " << mGlobLearnRate;
    KALDI_COUT << ", inside the components: ";
    for(size_t i=0; i<mNnet.size(); i++) {
      if(mNnet[i]->IsUpdatable()) {
        KALDI_COUT << " " << dynamic_cast<CuUpdatableComponent*>(mNnet[i])->LearnRate();
      }
    }
    KALDI_COUT << "\n" << std::flush;
  }



  void
  CuNetwork::
  SetMomentum(BaseFloat momentum)
  {
    LayeredType::iterator it;
    for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
      if((*it)->IsUpdatable()) {
        dynamic_cast<CuUpdatableComponent*>(*it)->Momentum(momentum);
      }
    }
  }

  void
  CuNetwork::
  SetWeightcost(BaseFloat weightcost)
  {
    LayeredType::iterator it;
    for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
      if((*it)->IsUpdatable()) {
        dynamic_cast<CuUpdatableComponent*>(*it)->Weightcost(weightcost);
      }
    }
  }

  void
  CuNetwork::
  SetL1(BaseFloat l1)
  {
    LayeredType::iterator it;
    for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
      if((*it)->GetType() == CuComponent::SPARSE_LINEARITY) {
        dynamic_cast<CuSparseLinearity*>(*it)->L1(l1);
      }
    }
  }
   

  CuComponent*
  CuNetwork::
  ComponentFactory(std::istream& rIn)
  {
    rIn >> std::ws;
    if(rIn.eof()) return NULL;

    CuComponent* pRet=NULL;
    CuComponent* pPred=NULL;

    std::string componentTag;
    size_t nInputs, nOutputs;

    rIn >> std::ws;
    rIn >> componentTag;
    if(componentTag == "") return NULL; //nothing left in the file

    //make it lowercase
    std::transform(componentTag.begin(), componentTag.end(), 
                   componentTag.begin(), tolower);

    if(componentTag[0] != '<' || componentTag[componentTag.size()-1] != '>') {
      KALDI_ERR << "Invalid component tag: " << componentTag;
    }

    //the 'endblock' tag terminates the network
    if(componentTag == "<endblock>") return NULL;

    rIn >> std::ws;
    rIn >> nOutputs;
    rIn >> std::ws;
    rIn >> nInputs;
    assert(nInputs > 0 && nOutputs > 0);

    //make coupling with predecessor
    if(mNnet.size() != 0) {
      pPred = mNnet.back();
    }
    
    //array with list of component tags
    static const std::string TAGS[] = {
      "<biasedlinearity>",
      "<blockdiagonallinearity>",
      "<sharedlinearity>",
      "<sparselinearity>",
      "<rbm>",
      "<rbmsparse>",

      "<softmax>",
      "<blocksoftmax>",
      "<sigmoid>",

      "<expand>",
      "<copy>",
      "<transpose>",
      "<blocklinearity>",
      "<bias>",
      "<window>",
      "<log>", 

      "<blockarray>",
    };

    static const int n_tags = sizeof(TAGS) / sizeof(TAGS[0]);
    int i;
    for(i=0; i<n_tags; i++) {
      if(componentTag == TAGS[i]) break;
    }
       
    //switch according to position in array TAGS
    switch(i) {
      case 0: pRet = new CuBiasedLinearity(nInputs,nOutputs,pPred); break;
      case 1: pRet = new CuBlockdiagonalLinearity(nInputs,nOutputs,pPred); break;
      case 2: pRet = new CuSharedLinearity(nInputs,nOutputs,pPred); break;
      case 3: pRet = new CuSparseLinearity(nInputs,nOutputs,pPred); break;
      case 4: pRet = new CuRbm(nInputs,nOutputs,pPred); break;
      case 5: pRet = new CuRbmSparse(nInputs,nOutputs,pPred); break;

      case 6: pRet = new CuSoftmax(nInputs,nOutputs,pPred); break;
      case 7: pRet = new CuBlockSoftmax(nInputs,nOutputs,pPred); break;
      case 8: pRet = new CuSigmoid(nInputs,nOutputs,pPred); break;

      case 9: pRet = new CuExpand(nInputs,nOutputs,pPred); break;
      case 10: pRet = new CuCopy(nInputs,nOutputs,pPred); break;
      case 11: pRet = new CuTranspose(nInputs,nOutputs,pPred); break;
      case 12: pRet = new CuBlockLinearity(nInputs,nOutputs,pPred); break;
      case 13: pRet = new CuBias(nInputs,nOutputs,pPred); break;
      case 14: pRet = new CuWindow(nInputs,nOutputs,pPred); break;
      case 15: pRet = new CuLog(nInputs,nOutputs,pPred); break;
     
      case 16: pRet = new CuBlockArray(nInputs,nOutputs,pPred); break;
      
      default: KALDI_ERR << "Unknown Component tag: " << componentTag;
    }
   
    //read components content
    pRet->ReadFromStream(rIn);
        
    //return
    return pRet;
  }


  void
  CuNetwork::
  ComponentDumper(std::ostream& rOut, CuComponent& rComp)
  {
    //use tags of all the components; or the identification codes
    //array with list of component tags
    static const CuComponent::ComponentType TYPES[] = {
      CuComponent::BIASED_LINEARITY,
      CuComponent::BLOCKDIAGONAL_LINEARITY,
      CuComponent::SHARED_LINEARITY,
      CuComponent::SPARSE_LINEARITY,
      CuComponent::RBM,
      CuComponent::RBM_SPARSE,

      CuComponent::SOFTMAX,
      CuComponent::BLOCK_SOFTMAX,
      CuComponent::SIGMOID,

      CuComponent::EXPAND,
      CuComponent::COPY,
      CuComponent::TRANSPOSE,
      CuComponent::BLOCK_LINEARITY,
      CuComponent::BIAS,
      CuComponent::WINDOW,
      CuComponent::LOG,

      CuComponent::BLOCK_ARRAY,
    };
    static const std::string TAGS[] = {
      "<biasedlinearity>",
      "<blockdiagonallinearity>",
      "<sharedlinearity>",
      "<sparselinearity>",
      "<rbm>",
      "<rbmsparse>",

      "<softmax>",
      "<blocksoftmax>",
      "<sigmoid>",

      "<expand>",
      "<copy>",
      "<transpose>",
      "<blocklinearity>",
      "<bias>",
      "<window>",
      "<log>",

      "<blockarray>",
    };
    static const int MAX = sizeof TYPES / sizeof TYPES[0];

    int i;
    for(i=0; i<MAX; ++i) {
      if(TYPES[i] == rComp.GetType()) break;
    }
    if(i == MAX) KALDI_ERR << "Unknown ComponentType";
    
    //dump the component tag
    rOut << TAGS[i] << " " 
         << rComp.GetNOutputs() << " " 
         << rComp.GetNInputs() << std::endl;

    //write components content
    rComp.WriteToStream(rOut);
  }


  
} //namespace

