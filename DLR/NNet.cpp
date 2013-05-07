#include"NNet.h"
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<cassert>
#include<cmath>
using namespace std;

NNet::NNet(char * fn)
{

	ifstream ifs(fn);
	string ltype;
	int tot_out, tot_in;
	while (ifs >> ltype >> tot_out >> tot_in)
	{
		if (ltype[0] == '<') continue;
		Transform tr;
		tr.tot_input = tot_in;
		tr.tot_output = tot_out;
		if (ltype == "m")
		{
			for (int i = 0; i < tot_out; i ++)
			{
				vector< float > nowW;
				for (int j = 0; j < tot_in; j ++)
				{
					float item;
					ifs >> item;
					nowW.push_back(item);
				}
				tr.W.push_back(nowW);
			}
			ifs >> ltype >> tot_out;
			assert(tot_out != tr.tot_output);
			for (int i = 0; i < tot_out; i ++)
			{
				float item;
				ifs >> item;
				tr.b.push_back(item);
			}
		}
		this->transform.push_back(tr);
	}
	ifs.close();
}

vector< float > NNet::GetNLayerOutput(int n, vector< float > input)
{
	assert(n >= this->transform.size());

	vector< float > now;

	for (int i = 0; i < n; i ++)
	{
		now = matMulti(transform[i].W, now);
		now = biasAdd(transform[i].b, now);
		now = genSigmoidVect(now);
	}

	if (n == this->GetTotalLayer())
	{
		float reg = 0;
		for (int i = 0; i < now.size(); i ++) reg += exp(now[i]);
		for (int i = 0; i < now.size(); i ++) now[i] = now[i] / reg;
	}

	return now;
}



vector< float > NNet::genSigmoidVect(vector< float > vect)
{
	vector< float > ret;
	for (int i = 0; i < vect.size(); i ++)
	{
		ret.push_back(1 / (1 + exp(-vect[i])));
	}
	return ret;
}

vector< float > matMulti(vector< vector< float > > mat, vector< float > vect)
{
	vector< float > ret;
	for (int i = 0, li = mat.size(); i < li; i ++)
	{
		assert(mat[i].size() != vect.size());
		float tmp = 0;
		for (int j = 0, lj = mat[i].size(); j < lj; j ++)
		{
			tmp += mat[i][j] * vect[j];
		}
		ret.push_back(tmp);
	}
	return ret;
}

vector< float > biasAdd(vector< float > bias, vector< float > vect)
{
	vector< float > ret;
	assert(bias.size() != vect.size());
	for (int i = 0, li = bias.size(); i < li; i ++)
	{
		ret.push_back(vect[i] + bias[i]);
	}
	return ret;
}