#include "Labels.h"
#include "Timer.h"


namespace TNet {


  ////////////////////////////////////////////////////////////////////////
  // Class LabelRepository::
  void
  LabelRepository::
  Init(const char* pLabelMlfFile, const char* pOutputLabelMapFile, const char* pLabelDir, const char* pLabelExt)
  {
    assert(NULL != pLabelMlfFile);
    assert(NULL != pOutputLabelMapFile);

    // initialize the label streams
    delete mpLabelStream; //if NULL, does nothing
    delete _mpLabelStream;
    _mpLabelStream = new std::ifstream(pLabelMlfFile);
    mpLabelStream  = new IMlfStream(*_mpLabelStream);

    // Label stream is initialized, just test it
    if(!mpLabelStream->good()) 
      KALDI_ERR << "Cannot open Label MLF file: " << pLabelMlfFile;

    // Index the labels (good for randomized file lists)
    Timer tim; tim.Start();
    mpLabelStream->Index();
    tim.End(); mIndexTime += tim.Val(); 

    // Read the state-label to state-id map
    ReadOutputLabelMap(pOutputLabelMapFile);

    // Store the label dir/ext
    mpLabelDir = pLabelDir;
    mpLabelExt = pLabelExt;

    // flag that initialization was completed
    mIsReady = true;
  }


  bool 
  LabelRepository::
  GenDesiredMatrix(BfMatrix& rDesired, size_t nFrames, size_t sourceRate, const char* pFeatureLogical)
  {
    //timer
    Timer tim; tim.Start();
    
    //Get the MLF stream reference...
    IMlfStream& mLabelStream = *mpLabelStream;
    //Build the file name of the label
    MakeHtkFileName(mpLabelFile, pFeatureLogical, mpLabelDir, mpLabelExt);

    //Find block in MLF file
    mLabelStream.Open(mpLabelFile);
    if(!mLabelStream.good()) {
      KALDI_WARN << "Cannot open label MLF record: " << mpLabelFile;
      mLabelStream.Close();
      return false;
    }

    //prepare a vector with desired matrix indices
    std::vector<size_t> tgt_id_vec;
    tgt_id_vec.reserve(nFrames);

    //aux variables
    std::string line, state;
    unsigned long long beg, end;
    size_t state_index;
    TagToIdMap::iterator it;
    
    //parse the label file, fill the vector tgt_id_vec
    while(!mLabelStream.eof()) {
      std::getline(mLabelStream, line);
      if(line == "") continue; //skip newlines/comments from MLF
      if(line[0] == '#') continue;

      std::istringstream& iss = mGenDesiredMatrixStream;
      iss.clear();
      iss.str(line);

      //parse the line
      //begin
      iss >> std::ws >> beg;
      if(iss.fail()) { 
        KALDI_ERR << "Cannot parse column 1 (begin)\n"
                  << "line: " << line << "\n"
                  << "file: " << mpLabelFile << "\n";
      }
      //end
      iss >> std::ws >> end;
      if(iss.fail()) { 
        KALDI_ERR << "Cannot parse column 2 (end)\n"
                  << "line: " << line << "\n"
                  << "file: " << mpLabelFile << "\n";
      }
      //state tag
      iss >> std::ws >> state;
      if(iss.fail()) { 
        KALDI_ERR << "Cannot parse column 3 (state_tag)\n"
                  << "line: " << line << "\n"
                  << "file: " << mpLabelFile << "\n";
      }

      //round up the begin/end times
      beg = (beg+sourceRate/2)/sourceRate;
      end = (end+sourceRate/2)/sourceRate; 
      
      //find the state id
      it = mLabelMap.find(state);
      if(mLabelMap.end() == it) {
        KALDI_ERR << "Unknown state tag: '" << state << "' file:'" << mpLabelFile;
      }
      state_index = it->second;

      //check that 'beg' time corresponds with number of elements already in 'tgt_id_vec'
      if(!beg == tgt_id_vec.size()) {
        KALDI_WARN << "Frame gap in the labels, skipping file : " << mpLabelFile;
        mLabelStream.Close();
        return false;
      }
      //fill the vector of ids'
      tgt_id_vec.insert(tgt_id_vec.end(),end-beg,state_index);
    }
    //close the label stream
    mLabelStream.Close();


    //may be too few/too much frames, tolerate +/- 10 frame difference
    if(tgt_id_vec.size() != nFrames) {
      char message[2048];
      if((tgt_id_vec.size() < nFrames) && (nFrames - tgt_id_vec.size() <= 10)) {
        //tolerate labels shorter by up tp 10 frames, fill with last tgt_id...
        size_t extra_frames = nFrames - tgt_id_vec.size();
        sprintf(message,"Filling extra %d frames of : %s in %s", extra_frames, state.c_str(), mpLabelFile);
        KALDI_WARN << message;
        tgt_id_vec.insert(tgt_id_vec.end(), nFrames-tgt_id_vec.size(), tgt_id_vec.back());
      } else if ((tgt_id_vec.size() > nFrames) && (tgt_id_vec.size() - nFrames <= 10))  {
        //tolerate labels longer by up to 10 frames 
        size_t extra_frames = tgt_id_vec.size()-nFrames;
        sprintf(message,"Labels longer than features by %d frames at : %s , truncating...", extra_frames, mpLabelFile); 
        KALDI_WARN << message;
      } else {
        //better skip that file
        sprintf(message,"Non-matching length of features %d and labels %d at : %s , skipping...", nFrames, tgt_id_vec.size(), mpLabelFile); 
        KALDI_WARN << message;
        return false;
      }
    }

    //resize the output matrix
    rDesired.Init(nFrames, mLabelMap.size(), true); //true: Zero()
    //fill the matrix with ones
    for(int r=0; r<rDesired.Rows(); r++) {
      rDesired(r,tgt_id_vec[r]) = 1.0;
    }

    //timer
    tim.End(); mGenDesiredMatrixTime += tim.Val();
    
    return true;
  }




  

  void
  LabelRepository::
  ReadOutputLabelMap(const char* file)
  {
    assert(mLabelMap.size() == 0);
    int i = 0;
    std::string state_tag;
    std::ifstream in(file);
    if(!in.good())
      KALDI_ERR << "Cannot open OutputLabelMapFile: " << file;

    in >> std::ws;
    while(!in.eof()) {
      in >> state_tag;
      in >> std::ws;
      assert(mLabelMap.find(state_tag) == mLabelMap.end());
      mLabelMap[state_tag] = i++;
    }

    in.close();
    assert(mLabelMap.size() > 0);
  }


}//namespace
