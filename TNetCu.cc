
/***************************************************************************
 *   copyright            : (C) 2011 by Karel Vesely,UPGM,FIT,VUT,Brno     *
 *   email                : iveselyk@fit.vutbr.cz                          *
 ***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the APACHE License as published by the          *
 *   Apache Software Foundation; either version 2.0 of the License,        *
 *   or (at your option) any later version.                                *
 *                                                                         *
 ***************************************************************************/

#define SVN_DATE       "$Date: 2013-01-15 20:05:30 +0100 (Tue, 15 Jan 2013) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 145 $"
#define SVN_ID         "$Id: TNetCu.cc 145 2013-01-15 19:05:30Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID  



/*** TNetLib includes */
#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "Labels.h"
#include "Common.h"
#include "MlfStream.h"
#include "UserInterface.h"
#include "Timer.h"

/*** TNet includes */
#include "cuObjectiveFunction.h"
#include "cuNnet.h"
#include "cuCache.h"

/*** STL includes */
#include <iostream>
#include <sstream>
#include <numeric>




//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TNET"

using namespace TNet;

void usage(const char* progname) 
{
  const char *tchrptr;
  if ((tchrptr = strrchr(progname, '\\')) != NULL) progname = tchrptr+1;
  if ((tchrptr = strrchr(progname, '/')) != NULL) progname = tchrptr+1;
  fprintf(stderr,
"\n%s version " MODULE_VERSION "\n"
"\nUSAGE: %s [options] DataFiles...\n\n"
" Option                                                     Default\n\n"
" -c         Enable crossvalidation                          off\n"
" -m file    Set label map of NN outputs                     \n"
" -n f       Set learning rate to f                          0.06\n"
" -o ext     Set target model ext                            None\n"
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -H mmf     Load NN macro file                              \n"
" -I mlf     Load master label file mlf                      \n"
" -L dir     Set input label (or net) dir                    Current\n"
" -M dir     Dir to write NN macro files                     Current\n"
" -O fn      Objective function [mse,xent]                   xent\n"
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n" 
" -V         Print version information                       Off\n"
" -X ext     Set input label file ext                        lab\n"
"\n"
"BUNCHSIZE CACHESIZE CROSSVALIDATE FEATURETRANSFORM L1 LEARNINGRATE LEARNRATEFACTORS MLFTRANSC MOMENTUM NATURALREADORDER OBJECTIVEFUNCTION OUTPUTLABELMAP PRINTCONFIG PRINTVERSION RANDOMIZE SCRIPT SEED SOURCEMLF SOURCEMMF SOURCETRANSCDIR SOURCETRANSCEXT TARGETMMF TARGETMODELDIR TARGETMODELEXT TRACE USEGPUID WEIGHTCOST\n"
"\n"
"STARTFRMEXT ENDFRMEXT CMEANDIR CMEANMASK VARSCALEDIR VARSCALEMASK VARSCALEFN TARGETKIND DERIVWINDOWS DELTAWINDOW ACCWINDOW THIRDWINDOW\n"
"\n"
" %s is Copyright (C) 2010-2011 Karel Vesely\n"
" licensed under the APACHE License, version 2.0\n"
" Bug reports, feedback, etc, to: iveselyk@fit.vutbr.cz\n"
"\n", progname, progname, progname);
  exit(-1);
}




///////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
//


int main(int argc, char *argv[]) try
{
  const char* p_option_string =
    " -c n   CROSSVALIDATE=TRUE"
    " -m r   OUTPUTLABELMAP" 
    " -n r   LEARNINGRATE" 
    " -o r   TARGETMODELEXT" 
    " -D n   PRINTCONFIG=TRUE"
    " -H l   SOURCEMMF"
    " -I r   SOURCEMLF"
    " -L r   SOURCETRANSCDIR"
    " -M r   TARGETMODELDIR"
    " -O r   OBJECTIVEFUNCTION" 
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE"
    " -X r   SOURCETRANSCEXT";


  UserInterface        ui;
  FeatureRepository    feature_repo;
  LabelRepository      label_repo;
  CuNetwork            network;
  CuNetwork            transform_network;
  CuObjectiveFunction* p_obj_function = NULL;
  Timer                timer;
  Timer                timer_frontend;
  double               time_frontend = 0.0;

 
  const char*                       p_script;
  const char*                       p_output_label_map;
  BaseFloat                         learning_rate;
  const char*                       learning_rate_factors;
  BaseFloat                         momentum;
  BaseFloat                         weightcost;
  BaseFloat                         l1;
  CuObjectiveFunction::ObjFunType   obj_fun_id;

  const char*                       p_source_mmf_file;
  const char*                       p_input_transform;
  //const char*                       p_input_transform2;

  const char*                       p_targetmmf; //< SNet legacy --TARGETMMF
        char                        p_trg_mmf_file[4096];
  const char*                       p_trg_mmf_dir;
  const char*                       p_trg_mmf_ext;

  const char*                       p_source_mlf_file;
  const char*                       p_src_lbl_dir;
  const char*                       p_src_lbl_ext;

  int                               bunch_size;
  int                               cache_size;
  bool                              randomize;
  long int                          seed;

  bool                              cross_validate;

  int                               trace;

  int                               gpu_select;

  // variables for feature repository
  bool                              swap_features;
  int                               target_kind;
  int                               deriv_order;
  int*                              p_deriv_win_lenghts;
  int                               start_frm_ext;
  int                               end_frm_ext;
        char*                       cmn_path;
        char*                       cmn_file;
  const char*                       cmn_mask;
        char*                       cvn_path;
        char*                       cvn_file;
  const char*                       cvn_mask;
  const char*                       cvg_file;

 
  // OPTION PARSING ..........................................................
  // use the STK option parsing
  if (argc == 1) { usage(argv[0]); return 1; }
  int args_parsed = ui.ParseOptions(argc, argv, p_option_string, SNAME);


  // OPTION RETRIEVAL ........................................................
  // extract the feature parameters
  swap_features = !ui.GetBool(SNAME":NATURALREADORDER", TNet::IsBigEndian());
  
  target_kind = ui.GetFeatureParams(&deriv_order, &p_deriv_win_lenghts,
       &start_frm_ext, &end_frm_ext, &cmn_path, &cmn_file, &cmn_mask,
       &cvn_path, &cvn_file, &cvn_mask, &cvg_file, SNAME":", 0);


  // extract other parameters
  p_source_mmf_file   = ui.GetStr(SNAME":SOURCEMMF",     NULL);
  p_input_transform   = ui.GetStr(SNAME":FEATURETRANSFORM",  NULL);
  
  p_targetmmf         = ui.GetStr(SNAME":TARGETMMF",     NULL);//< has higher priority than "dir/file.ext" composition (SNet legacy)
  p_trg_mmf_dir       = ui.GetStr(SNAME":TARGETMODELDIR",  "");//< dir for composition
  p_trg_mmf_ext       = ui.GetStr(SNAME":TARGETMODELEXT",  "");//< ext for composition

  p_script            = ui.GetStr(SNAME":SCRIPT",         NULL);
  p_output_label_map  = ui.GetStr(SNAME":OUTPUTLABELMAP", NULL);
  learning_rate       = ui.GetFlt(SNAME":LEARNINGRATE"  , 0.06f);
  learning_rate_factors = ui.GetStr(SNAME":LEARNRATEFACTORS", NULL);
  momentum            = ui.GetFlt(SNAME":MOMENTUM"      , 0.0);
  weightcost          = ui.GetFlt(SNAME":WEIGHTCOST"    , 0.0);
  l1                  = ui.GetFlt(SNAME":L1"            , 0.0);

  obj_fun_id          = static_cast<CuObjectiveFunction::ObjFunType>(
                        ui.GetEnum(SNAME":OBJECTIVEFUNCTION", 
                                   CuObjectiveFunction::CROSS_ENTROPY, //< default
                                   "xent", CuObjectiveFunction::CROSS_ENTROPY,
                                   "mse", CuObjectiveFunction::MEAN_SQUARE_ERROR
                        ));

  p_source_mlf_file   = ui.GetStr(SNAME":SOURCEMLF",       NULL);
  p_src_lbl_dir       = ui.GetStr(SNAME":SOURCETRANSCDIR", NULL);
  p_src_lbl_ext       = ui.GetStr(SNAME":SOURCETRANSCEXT", "lab");


  bunch_size          = ui.GetInt(SNAME":BUNCHSIZE", 256);
  cache_size          = ui.GetInt(SNAME":CACHESIZE", 12800);
  randomize           = ui.GetBool(SNAME":RANDOMIZE", true);

  //cannot get long int
  seed                = ui.GetInt(SNAME":SEED", 0);

  cross_validate      = ui.GetBool(SNAME":CROSSVALIDATE",  false);

  trace       = ui.GetInt(SNAME":TRACE",               0);
  if(trace&4) { CuDevice::Instantiate().Verbose(true); }

  gpu_select  = ui.GetInt(SNAME":USEGPUID", -1);
  if(gpu_select >= 0) { CuDevice::Instantiate().SelectGPU(gpu_select); }



  // process the parameters
  if(ui.GetBool(SNAME":PRINTCONFIG", false)) {
    KALDI_COUT << std::endl;
    ui.PrintConfig(KALDI_COUT);
    KALDI_COUT << std::endl;
  }
  if(ui.GetBool(SNAME":PRINTVERSION", false)) {
    KALDI_COUT << std::endl;
    KALDI_COUT << "======= TNET v"MODULE_VERSION" =======" << std::endl;
    KALDI_COUT << std::endl;
  }
  ui.CheckCommandLineParamUse();
  

  // the rest of the parameters are the feature files
  for (; args_parsed < argc; args_parsed++) {
    feature_repo.AddFile(argv[args_parsed]);
  }

  //**************************************************************************
  //**************************************************************************
  // OPTION PARSING DONE .....................................................


  //read the input transform network
  if(NULL != p_input_transform) { 
    if(trace&1) KALDI_LOG << "Reading input transform network: " << p_input_transform;
    transform_network.ReadNetwork(p_input_transform);
  }


  //read the neural network
  if(NULL != p_source_mmf_file) { 
    if(trace&1) KALDI_LOG << "Reading network: " << p_source_mmf_file;
    network.ReadNetwork(p_source_mmf_file);
  } else {
    KALDI_ERR << "Source MMF must be specified [-H]";
  }


  // initialize the feature repository 
  feature_repo.Init(
    swap_features, start_frm_ext, end_frm_ext, target_kind,
    deriv_order, p_deriv_win_lenghts, 
    cmn_path, cmn_mask, cvn_path, cvn_mask, cvg_file
  );
  feature_repo.Trace(trace);
  if(NULL != p_script) {
    feature_repo.AddFileList(p_script);
  } else {
    KALDI_WARN << "The script file is missing [-S]";
  }


  // initialize the label repository
  if(NULL == p_source_mlf_file)
    KALDI_WARN << "Source mlf file file is missing [-I]";
  if(NULL == p_output_label_map)
    KALDI_WARN << "Output label map is missing [-m]";

  if(NULL != p_source_mlf_file && NULL != p_output_label_map) {
    if(trace&1) KALDI_LOG << "Indexing labels: " << p_source_mlf_file;
    label_repo.Init(p_source_mlf_file, p_output_label_map, p_src_lbl_dir, p_src_lbl_ext);
    label_repo.Trace(trace);
  } else if (NULL == p_source_mlf_file && NULL == p_output_label_map) {
    KALDI_LOG << "Using input/target pairs from : " << p_script << " for training";
  } else {
    KALDI_ERR << "Use both -m LABLIST -I MLF or non of these (input/target pair mode)";
  }


  //get objective function instance
  p_obj_function = CuObjectiveFunction::Factory(obj_fun_id);

  //set the learnrate, momentum, weightcost
  network.SetLearnRate(learning_rate, learning_rate_factors);
  network.SetMomentum(momentum);
  network.SetWeightcost(weightcost);
  network.SetL1(l1);

  //seed the random number generator
  if(seed == 0) {
    struct timeval tv;
    if (gettimeofday(&tv, 0) == -1) {
      assert(0 && "gettimeofday does not work.");
      exit(-1);
    }
    seed = (int)(tv.tv_sec) + (int)tv.tv_usec;
  }
  srand48(seed);


  
  
  //**********************************************************************
  //**********************************************************************
  // INITIALIZATION DONE .................................................
  //
  // Start training
  timer.Start();
  KALDI_COUT << "===== TNET " 
            << (cross_validate?"CROSSVALIDATION":"TRAINING") 
            << " STARTED =====" << std::endl;
  KALDI_COUT << "Objective function: " 
            << p_obj_function->GetTypeLabel() << std::endl;
  if(!cross_validate) {
    network.PrintLearnRate();
    KALDI_COUT << "momentum: " << momentum
              << " weightcost: " << weightcost << std::endl;
    KALDI_COUT << "using seed: " << seed << std::endl;
  }

  //make the cachesize divisible by bunchsize
  cache_size = (cache_size/bunch_size)*bunch_size;
  KALDI_COUT << "Bunchsize:" << bunch_size
            << " Cachesize:" << cache_size << "\n";

  CuCache cache;
  cache.Init(cache_size,bunch_size);
  cache.Trace(trace);
  feature_repo.Rewind();
  
  //**********************************************************************
  //**********************************************************************
  // MAIN LOOP
  //
  int feats_with_missing_labels = 0;
  CuMatrix<BaseFloat> feats, output, labs, globerr;
  while(!feature_repo.EndOfList()) {
    timer_frontend.Start();
    //fill cache
    while(!cache.Full() && !feature_repo.EndOfList()) {
      Matrix<BaseFloat> feats_host;
      CuMatrix<BaseFloat> feats_original;
      CuMatrix<BaseFloat> feats_expanded;

      //read feats, perfrom feature transform
      feature_repo.ReadFullMatrix(feats_host);
      feats_host.CheckData(feature_repo.Current().Logical());
      feats_original.CopyFrom(feats_host);
      
      //Feedforward block-wise (stable even if too long segments)...
      transform_network.Feedforward(feats_original,feats_expanded,start_frm_ext,end_frm_ext);

      //trim the start/end context
      int rows = feats_expanded.Rows()-start_frm_ext-end_frm_ext;
      CuMatrix<BaseFloat> feats_trim(rows,feats_expanded.Cols());
      feats_trim.CopyRows(rows,start_frm_ext,feats_expanded,0);

      //read labels
      Matrix<BaseFloat> labs_host; CuMatrix<BaseFloat> labs_cu;
      if(label_repo.IsReady()) {
        //read from label repository
        bool success = label_repo.GenDesiredMatrix(labs_host,feats_trim.Rows(), 
                                    feature_repo.CurrentHeader().mSamplePeriod,
                                    feature_repo.Current().Logical().c_str());
        if(!success) {
          feature_repo.MoveNext();
          feats_with_missing_labels++;
          continue;
        }
      } else {
        //we will use Feature/Target pairs from the FeatureRepository
        std::string feature_name = feature_repo.Current().Logical();
        //go to next file
        feature_repo.MoveNext();
        feature_repo.ReadFullMatrix(labs_host);
        //check the dim
        if(labs_cu.Rows() != feats_trim.Rows()) {
          KALDI_ERR << "Nonmatching number of rows,\n"
                    << "INPUT_ROWS=" << feats_trim.Rows() << " " << feature_name << "\n"
                    << "TARGET_ROWS=" << labs_host.Rows() << " " << feature_repo.Current().Logical(); 
        }
      }
      labs_cu.CopyFrom(labs_host);
      
      //add to cache
      cache.AddData(feats_trim,labs_cu);

      feature_repo.MoveNext();
    }
    timer_frontend.End(); time_frontend += timer_frontend.Val();
   
    if(randomize) { 
      //randomize the cache
      cache.Randomize();
    }

    while(!cache.Empty()) {
      //get training data
      cache.GetBunch(feats,labs);

      //forward pass
      network.Propagate(feats,output);
      //accumulate error, get global err
      p_obj_function->Evaluate(output,labs,globerr);

      //backward pass
      if(!cross_validate) {
        network.Backpropagate(globerr);
      }
      if(trace&2) KALDI_COUT << "." << std::flush; 
    }
  }



  //**********************************************************************
  //**********************************************************************
  // TRAINING FINISHED .................................................
  //
  // Let's store the network, report the log


  if(trace&1) KALDI_LOG << "Training finished";
  KALDI_COUT << "features with missing labels : " << feats_with_missing_labels << "\n"; 

  //write the network
  if(!cross_validate) {
    if (NULL != p_targetmmf) {
      if(trace&1) KALDI_LOG << "Writing network: " << p_targetmmf;
      network.WriteNetwork(p_targetmmf);
    } else {
      MakeHtkFileName(p_trg_mmf_file, p_source_mmf_file, p_trg_mmf_dir, p_trg_mmf_ext);
      if(trace&1) KALDI_LOG << "Writing network: " << p_trg_mmf_file;
      network.WriteNetwork(p_trg_mmf_file);
    }
  }

  timer.End();
  KALDI_COUT << "===== TNET "
            << (cross_validate?"CROSSVALIDATION":"TRAINING") 
            << " FINISHED ( " << timer.Val() << "s ) "
            << "[FPS:" << p_obj_function->GetFrames() / timer.Val() 
            << ",RT:" << 1.0f / (p_obj_function->GetFrames() / timer.Val() / 100.0f)
            << "] =====" << std::endl;

  //report objective function (accuracy, frame counts...)
  KALDI_COUT << "-- " << (cross_validate?"CV ":"TR ") << p_obj_function->Report();

  if(trace &4) {
    KALDI_COUT << "\n== PROFILE ==\nT-fe: " << time_frontend << std::endl;
  }
  
  return  0; ///finish OK

} catch (std::exception& rExc) {
  KALDI_CERR << "\nException thrown\n!!!!!!!" << std::endl;
  KALDI_CERR << rExc.what() << std::endl;
  return  1;
}
