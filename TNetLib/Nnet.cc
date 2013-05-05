
#include <algorithm>
//#include <locale>
#include <cctype>

#include "Nnet.h"
#include "CRBEDctFeat.h"
#include "BlockArray.h"

namespace TNet {




void Network::Feedforward(const Matrix<BaseFloat>& in, Matrix<BaseFloat>& out, 
                          size_t start_frm_ext, size_t end_frm_ext) {
  //empty network: copy input to output 
  if(mNnet.size() == 0) {
    if(out.Rows() != in.Rows() || out.Cols() != in.Cols()) {
      out.Init(in.Rows(),in.Cols());
    }
    out.Copy(in);
    return;
  }
  
  //short input: propagate in one block  
  if(in.Rows() < 5000) { 
    Propagate(in,out);
  } else {//long input: propagate per parts
    //initialize
    out.Init(in.Rows(),GetNOutputs());
    Matrix<BaseFloat> tmp_in, tmp_out;
    int done=0, block=1024;
    //propagate first part
    tmp_in.Init(block+end_frm_ext,in.Cols());
    tmp_in.Copy(in.Range(0,block+end_frm_ext,0,in.Cols()));
    Propagate(tmp_in,tmp_out);
    out.Range(0,block,0,tmp_out.Cols()).Copy(
      tmp_out.Range(0,block,0,tmp_out.Cols())
    );
    done += block;
    //propagate middle parts
    while((done+2*block) < in.Rows()) {
      tmp_in.Init(block+start_frm_ext+end_frm_ext,in.Cols());
      tmp_in.Copy(in.Range(done-start_frm_ext, block+start_frm_ext+end_frm_ext, 0,in.Cols()));      Propagate(tmp_in,tmp_out);
      out.Range(done,block,0,tmp_out.Cols()).Copy(
        tmp_out.Range(start_frm_ext,block,0,tmp_out.Cols())
      );
      done += block;
    }
    //propagate last part
    tmp_in.Init(in.Rows()-done+start_frm_ext,in.Cols());
    tmp_in.Copy(in.Range(done-start_frm_ext,in.Rows()-done+start_frm_ext,0,in.Cols()));
    Propagate(tmp_in,tmp_out);
    out.Range(done,out.Rows()-done,0,out.Cols()).Copy(
      tmp_out.Range(start_frm_ext,tmp_out.Rows()-start_frm_ext,0,tmp_out.Cols())   
    );

    done += tmp_out.Rows()-start_frm_ext;
    assert(done == out.Rows());
  }
}


void Network::Propagate(const Matrix<BaseFloat>& in, Matrix<BaseFloat>& out) {
  //empty network: copy input to output 
  if(mNnet.size() == 0) {
    if(out.Rows() != in.Rows() || out.Cols() != in.Cols()) {
      out.Init(in.Rows(),in.Cols());
    }
    out.Copy(in);
    return;
  }
  
  //this will keep pointer to matrix 'in', for backprop
  mNnet.front()->SetInput(in); 

  //propagate
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    (*it)->Propagate();
  }

  //copy the output matrix
  const Matrix<BaseFloat>& mat = mNnet.back()->GetOutput();
  if(out.Rows() != mat.Rows() || out.Cols() != mat.Cols()) {
    out.Init(mat.Rows(),mat.Cols());
  }
  out.Copy(mat);

}


void Network::Backpropagate(const Matrix<BaseFloat>& globerr) {
  //pass matrix to last component
  mNnet.back()->SetErrorInput(globerr);

  // back-propagation : reversed order,
  LayeredType::reverse_iterator it;
  for(it=mNnet.rbegin(); it!=mNnet.rend(); ++it) {
    //first component does not backpropagate error (no predecessors)
    if(*it != mNnet.front()) {
      (*it)->Backpropagate();
    }
    //compute gradient if updatable component
    if((*it)->IsUpdatable()) {
      UpdatableComponent& comp = dynamic_cast<UpdatableComponent&>(**it);
      comp.Gradient(); //compute gradient 
    }
  }
}


void Network::AccuGradient(const Network& src, int thr, int thrN) {
  LayeredType::iterator it;
  LayeredType::const_iterator it2;

  for(it=mNnet.begin(), it2=src.mNnet.begin(); it!=mNnet.end(); ++it,++it2) {
    if((*it)->IsUpdatable()) {
      UpdatableComponent& comp = dynamic_cast<UpdatableComponent&>(**it);
      const UpdatableComponent& comp2 = dynamic_cast<const UpdatableComponent&>(**it2);
      comp.AccuGradient(comp2,thr,thrN);
    }
  }
}


void Network::Update(int thr, int thrN) {
  LayeredType::iterator it;

  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    if((*it)->IsUpdatable()) {
      UpdatableComponent& comp = dynamic_cast<UpdatableComponent&>(**it);
      comp.Update(thr,thrN);
    }
  }
}


Network* Network::Clone() {
  Network* net = new Network;
  LayeredType::iterator it;
  for(it = mNnet.begin(); it != mNnet.end(); ++it) {
    //clone
    net->mNnet.push_back((*it)->Clone());
    //connect network
    if(net->mNnet.size() > 1) {
      Component* last = *(net->mNnet.end()-1);
      Component* prev = *(net->mNnet.end()-2);
      last->SetInput(prev->GetOutput());
      prev->SetErrorInput(last->GetErrorOutput());
    }
  }

  //copy the learning rate
  //net->SetLearnRate(GetLearnRate());

  return net;
}


void Network::ReadNetwork(const char* pSrc) {
  std::ifstream in(pSrc);
  if(!in.good()) {
    KALDI_ERR << "Cannot read model: " << pSrc;
  }
  ReadNetwork(in);
  in.close();
}

  

void Network::ReadNetwork(std::istream& rIn) {
  //get the network elements from a factory
  Component *pComp;
  while(NULL != (pComp = ComponentFactory(rIn))) 
    mNnet.push_back(pComp);
}


void Network::WriteNetwork(const char* pDst) {
  std::ofstream out(pDst);
  if(!out.good()) {
    KALDI_ERR << "Cannot write model: " << pDst;
  }
  WriteNetwork(out);
  out.close();
}


void Network::WriteNetwork(std::ostream& rOut) {
  //dump all the componetns
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    ComponentDumper(rOut, **it);
  }
}


void
Network::
SetLearnRate(BaseFloat learnRate, const char* pLearnRateFactors) 
{
  //parse the learn rate factors: "0.1:0.5:0.6:1.0" to std::list
  std::list<BaseFloat> lr_factors;
  
  if(NULL != pLearnRateFactors) {
    //copy the factors to string
    std::string str(pLearnRateFactors);
    //replace any separator by ' '
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

  //set the learning rate
  BaseFloat lr_factor;
  LayeredType::iterator it;
  for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
    if((*it)->IsUpdatable()) {
      if(NULL != pLearnRateFactors) {
        lr_factor = lr_factors.front();
        lr_factors.pop_front();
      } else {
        lr_factor = 1.0;
      }
      dynamic_cast<UpdatableComponent*>(*it)->LearnRate(learnRate * lr_factor);
    }
  }

  //store the unscaled learning rate
  mGlobLearnRate = learnRate;
}



void
Network::
PrintLearnRate()
{
  assert(mNnet.size() > 0);
  KALDI_COUT << "Learning rate: global " << mGlobLearnRate;
  KALDI_COUT << ", inside the components: ";
  for(size_t i=0; i<mNnet.size(); i++) {
    if(mNnet[i]->IsUpdatable()) {
      KALDI_COUT << " " << dynamic_cast<UpdatableComponent*>(mNnet[i])->LearnRate();
    }
  }
  KALDI_COUT << "\n" << std::flush;
}





 

Component*
Network::
ComponentFactory(std::istream& rIn)
{
  rIn >> std::ws;
  if(rIn.eof()) return NULL;

  Component* pRet=NULL;
  Component* pPred=NULL;

  std::string componentTag;
  size_t nInputs, nOutputs;

  rIn >> std::ws;
  rIn >> componentTag;
  if(componentTag == "") return NULL; //nothing left in the file

  //make it lowercase
  std::transform(componentTag.begin(), componentTag.end(), 
                 componentTag.begin(), tolower);

  //the 'endblock' tag terminates the network
  if(componentTag == "<endblock>") return NULL;

  
  if(componentTag[0] != '<' || componentTag[componentTag.size()-1] != '>') {
    KALDI_ERR << "Invalid component tag:" << componentTag;
  }

  rIn >> std::ws;
  rIn >> nOutputs;
  rIn >> std::ws;
  rIn >> nInputs;
  assert(nInputs > 0 && nOutputs > 0);

  //make coupling with predecessor
  if(mNnet.size() == 0) {
    pPred = NULL;
  } else {
    pPred = mNnet.back();
  }
  
  //array with list of component tags
  static const std::string TAGS[] = {
    "<biasedlinearity>",
    "<sharedlinearity>",
    
    "<sigmoid>",
    "<tanh>",
    "<softmax>",
    "<blocksoftmax>",

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
  int i = 0;
  for(i=0; i<n_tags; i++) {
    if(componentTag == TAGS[i]) break;
  }
  
  //switch according to position in array TAGS
  switch(i) {
    case 0: pRet = new BiasedLinearity(nInputs,nOutputs,pPred); break;
    case 1: pRet = new SharedLinearity(nInputs,nOutputs,pPred); break;

    case 2: pRet = new Sigmoid(nInputs,nOutputs,pPred); break;
    case 3: pRet = new Tanh(nInputs,nOutputs,pPred); break;
    case 4: pRet = new Softmax(nInputs,nOutputs,pPred); break;
    case 5: pRet = new BlockSoftmax(nInputs,nOutputs,pPred); break;

    case 6: pRet = new Expand(nInputs,nOutputs,pPred); break;
    case 7: pRet = new Copy(nInputs,nOutputs,pPred); break;
    case 8: pRet = new Transpose(nInputs,nOutputs,pPred); break;
    case 9: pRet = new BlockLinearity(nInputs,nOutputs,pPred); break;
    case 10: pRet = new Bias(nInputs,nOutputs,pPred); break;
    case 11: pRet = new Window(nInputs,nOutputs,pPred); break;
    case 12: pRet = new Log(nInputs,nOutputs,pPred); break;
    
    case 13: pRet = new BlockArray(nInputs,nOutputs,pPred); break;

    default: KALDI_ERR << "Unknown Component tag:" << componentTag;
  }
 
  //read params if it is updatable component
  pRet->ReadFromStream(rIn);
  //return
  return pRet;
}


void
Network::
ComponentDumper(std::ostream& rOut, Component& rComp)
{
  //use tags of all the components; or the identification codes
  //array with list of component tags
  static const Component::ComponentType TYPES[] = {
    Component::BIASED_LINEARITY,
    Component::SHARED_LINEARITY,
    
    Component::SIGMOID,
    Component::TANH,
    Component::SOFTMAX,
    Component::BLOCK_SOFTMAX,

    Component::EXPAND,
    Component::COPY,
    Component::TRANSPOSE,
    Component::BLOCK_LINEARITY,
    Component::BIAS,
    Component::WINDOW,
    Component::LOG,

    Component::BLOCK_ARRAY,
  };
  static const std::string TAGS[] = {
    "<biasedlinearity>",
    "<sharedlinearity>",

    "<sigmoid>",
    "<tanh>",
    "<softmax>",
    "<blocksoftmax>",

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

  //dump the parameters (if any)
  rComp.WriteToStream(rOut);
}



  
} //namespace

