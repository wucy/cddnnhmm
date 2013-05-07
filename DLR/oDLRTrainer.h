#ifndef _h_oDLRTrainer
#define _h_oDLRTrainer

#include<iostream>
#include<vector>
#include"NNet.h"
using namespace std;

class oDLRTrainer
{
public:

	oDLRTrainer(char * nnet_fn);

	struct oDLRResults
	{
		vector< float > M_diag_linear;
		vector< float > b_linear;

		vector< vector< float > > raw_W_L;
		vector< float > new_b_L;
	};

	
	oDLRResults Train(float eps);

	
private:
	NNet nnet;
	vector< vector< float > > input_vects;
	vector< int > labels;

	void Mb_linear_train(vector< float > & M, vector< float > & b);
	void L_Wb_b_only_train(vector< float > & b);
	float eps;
};

#endif