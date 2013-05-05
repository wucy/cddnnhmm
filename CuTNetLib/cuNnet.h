#ifndef _CUNETWORK_H_
#define _CUNETWORK_H_

#include "cuComponent.h"

#include "cuBiasedLinearity.h"
//#include "cuBlockLinearity.h"
//#include "cuBias.h"
//#include "cuWindow.h"

#include "cuActivation.h"

#include "cuCRBEDctFeat.h"

#include "Vector.h"

#include <vector>


namespace TNet {

  class CuNetwork
  {
    //////////////////////////////////////
    // Typedefs
    typedef std::vector<CuComponent*> LayeredType;
      
      //////////////////////////////////////
      // Disable copy construction, assignment and default constructor
    private:
      CuNetwork(CuNetwork&); 
      CuNetwork& operator=(CuNetwork&);
       
    public:
      CuNetwork() { }
      CuNetwork(std::istream& rIn); 
      ~CuNetwork();

      void AddLayer(CuComponent* layer);

      int Layers()
      { return mNnet.size(); }

      CuComponent& Layer(int i)
      { return *mNnet[i]; }

      /// forward the data to the ouptut (blockwise so it is less memory hungry)
      void Feedforward(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>& out,
                       size_t start_frm_ext, size_t end_frm_ext); 

      /// forward the data to the output
      void Propagate(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>& out);

      /// backpropagate the error while updating weights
      void Backpropagate(const CuMatrix<BaseFloat>& globerr); 

      void ReadNetwork(const char* pSrc);     ///< read the network from file
      void WriteNetwork(const char* pDst);    ///< write network to file

      void ReadNetwork(std::istream& rIn);    ///< read the network from stream
      void WriteNetwork(std::ostream& rOut);  ///< write network to stream

      size_t GetNInputs() const; ///< Dimensionality of the input features
      size_t GetNOutputs() const; ///< Dimensionality of the desired vectors

      /// set the learning rate
      void SetLearnRate(BaseFloat learnRate, const char* pLearnRateFactors = NULL); 
      BaseFloat GetLearnRate();  ///< get the learning rate value
      void PrintLearnRate();     ///< log the learning rate values

      void SetMomentum(BaseFloat momentum);
      void SetWeightcost(BaseFloat weightcost);
      void SetL1(BaseFloat l1);

      void SetGradDivFrm(bool div);


    private:
      /// Creates a component by reading from stream
      CuComponent* ComponentFactory(std::istream& In);
      /// Dumps component into a stream
      void ComponentDumper(std::ostream& rOut, CuComponent& rComp);



    private:
      LayeredType mNnet; ///< container with the network layers
      BaseFloat mGlobLearnRate; ///< The global (unscaled) learn rate of the network
      const char* mpLearnRateFactors; ///< The global (unscaled) learn rate of the network
      

    //friend class NetworkGenerator; //<< For generating networks...

  };
    

  //////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuNetwork::
  inline 
  CuNetwork::
  CuNetwork(std::istream& rSource)
    : mGlobLearnRate(0.0), mpLearnRateFactors(NULL)
  {
    ReadNetwork(rSource);
  }


  inline
  CuNetwork::
  ~CuNetwork()
  {
    //delete all the components
    LayeredType::iterator it;
    for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
      delete *it;
      *it = NULL;
    }
    mNnet.resize(0);
  }

  
  inline void 
  CuNetwork::
  AddLayer(CuComponent* layer)
  {
    if(mNnet.size() > 0) {
      if(GetNOutputs() != layer->GetNInputs()) {
        KALDI_ERR << "Nonmatching dims! network-out-dim:" << GetNOutputs()
                  << " component-in-dim:" << layer->GetNInputs();
      }
      layer->SetInput(mNnet.back()->GetOutput());
      mNnet.back()->SetErrorInput(layer->GetErrorOutput());
    }
    mNnet.push_back(layer);
  }


  inline void
  CuNetwork::
  Propagate(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>& out)
  {
    //empty network => copy input
    if(mNnet.size() == 0) { 
      out.CopyFrom(in); 
      return;
    }

    //check dims
    if(in.Cols() != GetNInputs()) {
      KALDI_ERR << "Nonmatching dims"
                << " data dim is: " << in.Cols() 
                << " network needs: " << GetNInputs();
    }
    mNnet.front()->SetInput(in);
    
    //propagate
    LayeredType::iterator it;
    for(it=mNnet.begin(); it!=mNnet.end(); ++it) {
      (*it)->Propagate();
    }

    //copy the output
    out.CopyFrom(mNnet.back()->GetOutput());
  }




  inline void 
  CuNetwork::
  Backpropagate(const CuMatrix<BaseFloat>& globerr) 
  {
    mNnet.back()->SetErrorInput(globerr);

    // back-propagation
    LayeredType::reverse_iterator it;
    for(it=mNnet.rbegin(); it!=mNnet.rend(); ++it) {
      //compute errors for preceding network components
      (*it)->Backpropagate();
      //update weights if updatable component
      if((*it)->IsUpdatable()) {
        CuUpdatableComponent& rComp = dynamic_cast<CuUpdatableComponent&>(**it); 
        rComp.Update();
      }
    }
  }

      
  inline size_t
  CuNetwork::
  GetNInputs() const
  {
    if(!mNnet.size() > 0) return 0;
    return mNnet.front()->GetNInputs();
  }


  inline size_t
  CuNetwork::
  GetNOutputs() const
  {
    if(!mNnet.size() > 0) return 0;
    return mNnet.back()->GetNOutputs();
  }





} //namespace

#endif


