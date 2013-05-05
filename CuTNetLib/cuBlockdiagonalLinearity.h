#ifndef _CUBLOCKDIAGONAL_LINEARITY_H_
#define _CUBLOCKDIAGONAL_LINEARITY_H_


#include "cuComponent.h"
#include "cumatrix.h"


#include "Matrix.h"
#include "Vector.h"

#include <vector>


namespace TNet {

  class CuBlockdiagonalLinearity : public CuUpdatableComponent
  {
    public:

      CuBlockdiagonalLinearity(size_t nInputs, size_t nOutputs, CuComponent *pPred); 
      ~CuBlockdiagonalLinearity();  
      
      ComponentType GetType() const;
      const char* GetName() const;

      void PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);
      void BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y);

      void Update();

      void ReadFromStream(std::istream& rIn);
      void WriteToStream(std::ostream& rOut);

    protected:
      std::vector<CuMatrix<BaseFloat> > mLinearity;  ///< Matrix with neuron weights
      CuVector<BaseFloat> mBias;       ///< Vector with biases

      std::vector<CuMatrix<BaseFloat> > mLinearityCorrection; ///< Matrix for linearity updates
      CuVector<BaseFloat> mBiasCorrection;      ///< Vector for bias updates

      size_t mNBlocks;

  };




  ////////////////////////////////////////////////////////////////////////////
  // INLINE FUNCTIONS 
  // CuBlockdiagonalLinearity::
  inline 
  CuBlockdiagonalLinearity::
  CuBlockdiagonalLinearity(size_t nInputs, size_t nOutputs, CuComponent *pPred)
    : CuUpdatableComponent(nInputs, nOutputs, pPred), 
      //mLinearity(nInputs,nOutputs), mBias(nOutputs),
      //mLinearityCorrection(nInputs,nOutputs), mBiasCorrection(nOutputs)
      mNBlocks(0)
  { 
    //mLinearityCorrection.SetConst(0.0);
    //mBiasCorrection.SetConst(0.0);
  }


  inline
  CuBlockdiagonalLinearity::
  ~CuBlockdiagonalLinearity()
  { }

  inline CuComponent::ComponentType
  CuBlockdiagonalLinearity::
  GetType() const
  {
    return CuComponent::BLOCKDIAGONAL_LINEARITY;
  }

  inline const char*
  CuBlockdiagonalLinearity::
  GetName() const
  {
    return "<blockdiagonallinearity>";
  }



} //namespace



#endif
