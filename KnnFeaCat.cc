
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

#define SVN_DATE       "$Date: 2012-01-27 16:33:21 +0100 (Fri, 27 Jan 2012) $"
#define SVN_AUTHOR     "$Author: iveselyk $"
#define SVN_REVISION   "$Revision: 98 $"
#define SVN_ID         "$Id: TFeaCat.cc 98 2012-01-27 15:33:21Z iveselyk $"

#define MODULE_VERSION "1.0.0 "__TIME__" "__DATE__" "SVN_ID 


 
#include "Error.h"
#include "Timer.h"
#include "Features.h"
#include "Common.h"
#include "UserInterface.h"

#include "Nnet.h"

#include <sstream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<cassert>
#include<vector>
#include<functional>
#include<queue>
#include<boost/thread/thread.hpp>
#include<boost/thread/mutex.hpp>


//////////////////////////////////////////////////////////////////////
// DEFINES
//

#define SNAME "TFEACAT"

using namespace TNet;


 

const int _FEATURE_SIZE = 429;
const long _TOTAL_TRAIN = 6639272;
const long _TOTAL_CV = 2528933;
const int _STATE_SIZE = 2077;
const int __CPU = 10;
const int ___K = 10;

float * train_set;
float * test_set;
int * train_label;
int * test_label;



struct loss_item {
	float * vect;
	float loss;
	int id;
	friend bool operator < (const loss_item & lhs, const loss_item & rhs) {
		return lhs.loss < rhs.loss;
	}
	loss_item(float l = -1, float * v = 0, int i = -1) {
		vect = v;
		loss = l;
		id = i;
	}
	loss_item(const loss_item & lit) {
		vect = lit.vect;
		loss = lit.loss;
		id = lit.id;
	}
};

int * label_init(string head, int col) {
	int * ret = new int[col];
	string name = head + ".label";
	FILE * file = fopen(name.c_str(), "r+");
	for (int i = 0; i < col; i ++) fscanf(file, "%d", &ret[i]);
	cout << head << " label finished." << endl;
	return ret;
}


float * data_init(string head, long col) {
	string fea_name = head + ".feature";
	FILE * fea_file = fopen(fea_name.c_str(), "rb");
	long total = col * _FEATURE_SIZE * sizeof(float);
	float * matrix = malloc(total);
	assert(matrix);
	long x = fread(matrix, sizeof(float), col * _FEATURE_SIZE, fea_file);
	if (x != col * _FEATURE_SIZE) {
		std::cerr << x << " error!" << endl;
	}
	fclose(fea_file);
	cout << head << " finished." << endl;
	return matrix;
}

void push_queue(int k, std::priority_queue<loss_item> & queue, int train_id, float * train_vect, float * my_vect) {
	double loss = 0;
	for (int i = 0; i < _FEATURE_SIZE; i ++) {
		loss += (train_vect[i] - my_vect[i]) * (train_vect[i] - my_vect[i]);
	}
	loss_item item(loss, train_vect, train_id);
	if (queue.size() < k) {
		queue.push(item);
	} else {
		if(item.loss < queue.top().loss) {
			queue.pop();
			queue.push(item);
		}
	}
}

boost::mutex mt;

struct sub_knn_thread {
	int sta;
	int end;
	float * train_set;
	float * my_vect;
	int k;
	std::priority_queue<loss_item> & main_queue;
	sub_knn_thread(int s, int e, int kk, float * ts, float * ms, std::priority_queue<loss_item> & mq):main_queue(mq) {
		sta = s;
		end = e;
		my_vect = ms;
		train_set = ts;
		k = kk;
	}
	void operator()() {
		//std::cerr << "coming! " << sta << "\t" << end << endl;
		std::priority_queue<loss_item> queue;
		for (int i = sta; i <= end; i ++) {
			float * tis = train_set + i * _FEATURE_SIZE;
			double loss = 0;
			for (int j = 0; j < _FEATURE_SIZE; j ++) {
				loss += (tis[j] - my_vect[j]) * (tis[j] - my_vect[j]);
			}
			loss_item item(loss, tis, train_label[i]);
			if (queue.size() < k) {
				queue.push(item);
			} else {
				if(item.loss < queue.top().loss) {
					queue.pop();
					queue.push(item);
				}
			}
		}
		
		while (queue.size()) {
			//cout << "!!!" << endl;
			boost::mutex::scoped_lock lock(mt);
			loss_item now = queue.top();
			queue.pop();
			if (main_queue.size() < k) {
				main_queue.push(now);
			} else {
				if(now.loss < main_queue.top().loss) {
					main_queue.pop();
					main_queue.push(now);
				}
			}
		}
	}
};

void knn(float * vect, int k, float * ret) {
	std::priority_queue<loss_item> queue;
	int stapos[__CPU] = {0};
	int endpos[__CPU] = {0};
	boost::thread_group grp;
	for (int i = 0; i < __CPU; i ++) {
		stapos[i] = _TOTAL_TRAIN / __CPU * i;
		endpos[i] = _TOTAL_TRAIN / __CPU * (i + 1) - 1;
		if (i == __CPU - 1) endpos[i] = _TOTAL_TRAIN;
		sub_knn_thread thr(stapos[i], endpos[i], k, train_set, vect, queue);
		grp.create_thread(thr);
	}
	grp.join_all();
	
	float * weight = new float[_STATE_SIZE];
	for (int i = 0; i < _STATE_SIZE; i ++) weight[i] = 0;
	float tot = 0;
	while (queue.size()) {
		loss_item now = queue.top();
		float delta = 1.0 / now.loss;
		weight[now.id] += delta; // now.loss;
		tot += delta;
		queue.pop();
	}
	for (int i = 0; i < _STATE_SIZE; i ++) {
		ret[i] = (weight[i] + 1e-6) / (tot + 1e-6 * _STATE_SIZE);
	}
	delete [] weight;
}


void gen_knn(Matrix<BaseFloat> & in, Matrix<BaseFloat> & out, int start_frm_ext, int end_frm_ext) {
	Matrix<BaseFloat> mat(in.Rows(), _STATE_SIZE);
	BaseFloat * vect = new BaseFloat[in.Cols()];
	BaseFloat * fea = new BaseFloat[_STATE_SIZE];
	for (int i = 0; i < in.Rows(); i ++) {
		for (int j = 0; j < in.Cols(); j ++) {
			vect[j] = in(i, j);
		}
		knn(vect, ___K, fea);
		for (int j = 0; j < _STATE_SIZE; j ++) {
			mat(i, j) = fea[j];
		}
	}
	delete [] vect;
	delete [] fea;
	out.Copy(mat);
}

void usage(const char* progname) 
{
  const char *tchrptr;
  if ((tchrptr = strrchr(progname, '\\')) != NULL) progname = tchrptr+1;
  if ((tchrptr = strrchr(progname, '/')) != NULL) progname = tchrptr+1;
  fprintf(stderr,
"\n%s version " MODULE_VERSION "\n"
"\nUSAGE: %s [options] DataFiles...\n\n"
":TODO:\n\n"
" Option                                                     Default\n\n"
" -l dir     Set target directory for features               Current\n"
" -y ext     Set target feature ext                          fea\n"
" -A         Print command line arguments                    Off\n" 
" -C cf      Set config file to cf                           Default\n"
" -D         Display configuration variables                 Off\n"
" -H mmf     Load NN macro file                              \n"  
" -S file    Set script file                                 None\n"
" -T N       Set trace flags to N                            0\n"
" -V         Print version information                       Off\n"
"\n"
"FEATURETRANSFORM GMMBYPASS LOGPOSTERIOR NATURALREADORDER PRINTCONFIG PRINTVERSION SCRIPT SOURCEMMF TARGETPARAMDIR TARGETPARAMEXT TRACE\n"
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

train_label = label_init("data/final/train", _TOTAL_TRAIN);
train_set = data_init("data/final/train", _TOTAL_TRAIN);

  const char* p_option_string =
    " -l r   TARGETPARAMDIR" 
    " -y r   TARGETPARAMEXT" 
    " -D n   PRINTCONFIG=TRUE"
    " -H l   SOURCEMMF"
    " -S l   SCRIPT"
    " -T r   TRACE"
    " -V n   PRINTVERSION=TRUE";

  if(argc == 1) { usage(argv[0]); }

  UserInterface        ui;
  FeatureRepository    feature_repo;
  Network              transform_network;
  Network              network;
  Timer                tim;

 
  const char*                       p_script;
        char                        p_target_fea[4096];
  const char*                       p_target_fea_dir;
  const char*                       p_target_fea_ext;

  const char*                       p_source_mmf_file;
  const char*                       p_input_transform;

  bool                              gmm_bypass;
  bool                              log_posterior;
  int                               trace;

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
  int ii = ui.ParseOptions(argc, argv, p_option_string, SNAME);


  // OPTION RETRIEVAL ........................................................
  // extract the feature parameters
  swap_features = !ui.GetBool(SNAME":NATURALREADORDER", TNet::IsBigEndian());
  
  target_kind = ui.GetFeatureParams(&deriv_order, &p_deriv_win_lenghts,
       &start_frm_ext, &end_frm_ext, &cmn_path, &cmn_file, &cmn_mask,
       &cvn_path, &cvn_file, &cvn_mask, &cvg_file, SNAME":", 0);


  // extract other parameters
  p_source_mmf_file   = ui.GetStr(SNAME":SOURCEMMF",     NULL);
  p_input_transform   = ui.GetStr(SNAME":FEATURETRANSFORM",  NULL);

  p_script            = ui.GetStr(SNAME":SCRIPT",         NULL);
  p_target_fea_dir    = ui.GetStr(SNAME":TARGETPARAMDIR", NULL);
  p_target_fea_ext    = ui.GetStr(SNAME":TARGETPARAMEXT", NULL);
  
  gmm_bypass          = ui.GetBool(SNAME":GMMBYPASS",     false);
  log_posterior       = ui.GetBool(SNAME":LOGPOSTERIOR",  false);
   
  trace               = ui.GetInt(SNAME":TRACE",          00);

  
  // process the parameters
  if(ui.GetBool(SNAME":PRINTVERSION", false)) {
    std::cout << "Version: "MODULE_VERSION"" << std::endl;
  }
  if(ui.GetBool(SNAME":PRINTCONFIG", false)) {
    std::cout << std::endl;
    ui.PrintConfig(std::cout);
    std::cout << std::endl;
  }
  ui.CheckCommandLineParamUse();
  

  // the rest of the parameters are the feature files
  for (; ii < argc; ii++) {
    feature_repo.AddFile(argv[ii]);
  }

  //**************************************************************************
  //**************************************************************************
  // OPTION PARSING DONE .....................................................

  //read the input transform network
  if(NULL != p_input_transform) { 
    if(trace&1) TraceLog(std::string("Reading input transform network: ")+p_input_transform);
    transform_network.ReadNetwork(p_input_transform);
  }



  //read the neural network
  if(NULL != p_source_mmf_file) { 
    if(trace&1) TraceLog(std::string("Reading network: ")+p_source_mmf_file);
    //network.ReadNetwork(p_source_mmf_file);
  } else {
    std::std::cerr << "Source MMF must be specified [-H]\n";
  }


  //initialize the FeatureRepository
  feature_repo.Init(
    swap_features, start_frm_ext, end_frm_ext, target_kind,
    deriv_order, p_deriv_win_lenghts, 
    cmn_path, cmn_mask, cvn_path, cvn_mask, cvg_file
  );
  if(NULL != p_script) {
    feature_repo.AddFileList(p_script);
  } 
  if(feature_repo.QueueSize() <= 0) {
    KALDI_ERR << "No input features specified,\n"
              << " try [-S SCP] or positional argument";
  }

  //**************************************************************************
  //**************************************************************************
  // MAIN LOOP ...............................................................

  //progress
  size_t cnt = 0;
  size_t step = feature_repo.QueueSize() / 100;
  if(step == 0) step = 1;
  tim.Start();

  //data carriers
  Matrix<BaseFloat> feats_in,feats_out,nnet_out;
  //process all the feature files
  for(feature_repo.Rewind(); !feature_repo.EndOfList(); feature_repo.MoveNext()) {
    //read file
    feature_repo.ReadFullMatrix(feats_in);

    //pass through transform network
    //transform_network.Propagate(feats_in, feats_out);
    transform_network.Feedforward(feats_in, feats_out, start_frm_ext, end_frm_ext);

	std::std::cerr << feats_out.Rows() << "\t" << feats_out.Cols() << std::endl;
    char x; std::cin >> x;
    //pass through network
    //network.Propagate(feats_out,nnet_out);
    gen_knn(feats_out,nnet_out,start_frm_ext,end_frm_ext);
    //network.Feedforward(feats_out,nnet_out,start_frm_ext,end_frm_ext); 
    //get the ouput, trim the start/end context
    feats_out.Init(nnet_out.Rows()-start_frm_ext-end_frm_ext,nnet_out.Cols());
    memcpy(feats_out.pData(),nnet_out.pRowData(start_frm_ext),feats_out.MSize());
   
    //GMM bypass for HVite using posteriors as features
    if(gmm_bypass) {
      for(size_t i=0; i<feats_out.Rows(); i++) {
        for(size_t j=0; j<feats_out.Cols(); j++) {
          feats_out(i,j) = static_cast<BaseFloat>(sqrt(-2.0*log(feats_out(i,j))));
        }
      }
    }
  
    //Convert posteriors to logdomain
    if(log_posterior) {
      for(size_t i=0; i<feats_out.Rows(); i++) {
        for(size_t j=0; j<feats_out.Cols(); j++) {
          feats_out(i,j) = static_cast<BaseFloat>(log(feats_out(i,j)));
        }
      }
    }

    //build filename
    MakeHtkFileName(p_target_fea, 
                    feature_repo.Current().Logical().c_str(),
                    p_target_fea_dir, p_target_fea_ext);
    //save output   
    int sample_period = feature_repo.CurrentHeader().mSamplePeriod;
    feature_repo.WriteFeatureMatrix(feats_out,p_target_fea,PARAMKIND_USER,sample_period);
    
    //progress
    if(trace&1) {
      if((cnt++ % step) == 0) std::cout << 100 * cnt / feature_repo.QueueSize() << "%, " << std::flush;
    }
  }
  
  //finish
  if(trace&1) {
    tim.End();
    std::cout << "TFeaCat finished: " << tim.Val() << "s" <<std::endl;
  }
  return 0;

} catch (std::exception& rExc) {
  std::std::cerr << "Exception thrown" << std::endl;
  std::std::cerr << rExc.what() << std::endl;
  return 1;
}
