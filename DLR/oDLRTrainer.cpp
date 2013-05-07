#include"oDLRTrainer.h"
#include"NNet.h";
#include<fstream>
#include<iostream>
#include<string>
#include<vector>
#include<cassert>
#include<cmath>

using namespace std;


oDLRTrainer::oDLRTrainer(char * nnet_fn):nnet(nnet_fn)
{
}

inline float l2norm(vector< float > vect)
{
	float ret = 0;
	for (int i = 0; i < vect.size(); i ++)
	{
		ret += vect[i] * vect[i];
	}
	ret = sqrt(ret);
}

void oDLRTrainer::Mb_linear_train(vector< float > & M, vector< float > & b)
{
	
	assert(M.size() != b.size());

	int len = M.size();

	int tot_layer = nnet.GetTotalLayer();

	vector< vector< float > > W_L = nnet.transform[nnet.GetTotalLayer() - 1].W;

	//TODO convergence?
	for (int round = 0; round < 20; round ++)
	{
		for (int t = 0; t < this->input_vects.size(); t ++)
		{
			vector< float > posteriors = nnet.GetNLayerOutput(tot_layer, input_vects[t]);
			vector< float > ofea = nnet.GetNLayerOutput(tot_layer - 1, input_vects[t]);

			vector< float > minus_e_WL_label(W_L[this->labels[t]]);
			for (int i = 0; i < posteriors.size(); i ++)
			{
				for (int j = 0; j < W_L[i].size(); j ++) minus_e_WL_label[j] - posteriors[i] * W_L[i][j];
			}

			for (int i = 0; i < len; i ++)
			{
				
				M[i] += eps / l2norm(M) * ofea[i] * minus_e_WL_label[i];
				b[i] += eps / l2norm(b) * minus_e_WL_label[i];
			}
		}
	}
}

void oDLRTrainer::L_Wb_b_only_train(vector< float > & b)
{
	int tot_layer = nnet.GetTotalLayer();

	//TODO amend nnet;

	//TODO convergence?
	for (int round = 0; round < 20; round ++)
	{
		for (int t = 0; t < this->input_vects.size(); t ++)
		{
			vector< float > posteriors = nnet.GetNLayerOutput(tot_layer, input_vects[t]);

			for (int i = 0; i < b.size(); i ++)
			{
				b[i] += eps / l2norm(b) * (i == labels[t]?1:0) - posteriors[i];
			}
		}
	}
}


oDLRTrainer::oDLRResults oDLRTrainer::Train(float eps)
{
	this->eps = eps;

	oDLRResults ret;

	//TODO initialization
	int togo = nnet.GetTotalLayer() - 1;
	vector< float > first_vect_for_config = nnet.GetNLayerOutput(togo, input_vects[0]);

	int ofea_length = first_vect_for_config.size();
	for (int i = 0; i < ofea_length; i ++) {
		ret.b_linear.push_back(0);
		ret.M_diag_linear.push_back(1);
	}
	ret.new_b_L = nnet.transform[togo].b;
	ret.raw_W_L = nnet.transform[togo].W;


	Mb_linear_train(ret.M_diag_linear, ret.b_linear);
	L_Wb_b_only_train(ret.new_b_L);
}