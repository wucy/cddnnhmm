#ifndef _h_NNet
#define _h_NNet

#include<iostream>
#include<vector>
using namespace std;

class NNet
{
	friend class oDLRTrainer;
public:
	NNet(char * fn);
	vector< float > GetNLayerOutput(int n, vector < float > input);
	int GetTotalLayer() { return transform.size(); }

protected:

	struct Transform
	{
		vector< vector< float > > W;
		vector < float > b;
		int tot_input, tot_output;
	};

	vector< Transform > transform;

	enum LayerType
	{
		LINEAR,
		SIGMOID,
		SOFTMAX,
	};

private:
	vector< float > genSigmoidVect(vector< float > vect);
};

#endif
