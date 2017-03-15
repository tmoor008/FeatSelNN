#include <iostream>
#include <string> 
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath> 
#include <algorithm>
#include <numeric>

using namespace std;

bool loadData(const string &infile, vector<vector<float>> &v, float &min, float &max)
{
	vector<float> tmp;
	ifstream fin;
	string line;
	float val;
	int ck = 1;
    
    fin.open(infile.c_str());
    
    if (!fin.is_open())
    {
        return false;
    }
    
	 while(std::getline(fin, line))
	 {
	 	istringstream s(line);
	 	while(s >> val)
	 	{
	 		if (ck == 1)
	 		{
	 			ck = 0;
	 		}
	 		else
	 		{
	 			if (val < min)
				{
					min = val;
				}
				if (val > max)
				{
					max = val;
				}
	 		}
	 		tmp.push_back(val);
	 	}
 		v.push_back(tmp);
 		tmp.clear();
 		ck = 1;
	 }
	 
	 //check that data is imported correctly
	 /*
	 for (unsigned i = 0; i < v.size(); ++i)
	 {
		 for (unsigned j = 0; j < v.at(i).size(); ++j)
		 {
		 	cout << v.at(i).at(j) << " ";
		 }
		 cout << endl;
	 }
	 */
	 
    fin.close();
    
    return true;
}

void normalize(vector<vector<float>> &v, float &min, float &max)
{
	float tmp;
	/*
	for (unsigned i = 0; i < v.size(); ++i)
	{
		for (unsigned j = 1; j < v.at(i).size(); ++j)
		{
			if (v.at(i).at(j) < min)
			{
				min = v.at(i).at(j);
			}
			if (v.at(i).at(j) > max)
			{
				max = v.at(i).at(j);
			}
		}
	}
	*/
	
	//cout << "Min: " << min << endl;
	//cout << "Max: " << max << endl;
	
	float denom = max - min;
	//cout << denom << endl;
	for (unsigned i = 0; i < v.size(); ++i)
	{
		for (unsigned j = 1; j < v.at(i).size(); ++j)
		{
			tmp = (v.at(i).at(j) - min) / denom;
			v.at(i).at(j) = tmp;   
			//cout << v.at(i).at(j) << " ";
		}
		//cout << endl;
	}
}

void display(vector<unsigned> &v)
{
	for (unsigned i = 0; i < v.size(); ++i)
	{
			cout << v.at(i);
			if (i != v.size() - 1)
			{
				cout << ",";
			}
	}
	//cout << endl;
}

bool isInFeatSub(unsigned &num, vector<unsigned> &featureSub)
{
	//cout << "size: " << featureSub.size() << endl;
	for (unsigned i = 0; i < featureSub.size(); ++i)
	{
		//cout << "Checking feature: " << featureSub.at(i) << endl;
		//cout << "num is : " << num << endl;
		if (num == featureSub.at(i))
		{
			//cout << "found" << endl;
			return true;
		}
	}
	return false;
}

int hlprNN(vector<unsigned> &featureSub, vector<vector<float>> &v, unsigned &indexExcl, bool allFlag)
{
	float closest, closeInd, sum, dist;
	sum = 0;
	closeInd = -1;
	
	closest = INFINITY;
	
	for(unsigned i = 0; i < v.size(); ++i)
	{
		//cout << "LOOP" << endl;
		if (i != indexExcl)
		{
		
			for (unsigned j = 1; j < v.at(i).size(); ++j)
			{
				if (isInFeatSub(j, featureSub) || allFlag == true)
				{
					//cout << "Excl val: " << v.at(indexExcl).at(j) << endl;
					//cout << "Diff pos, feat val: " << v.at(i).at(j) << endl;
					sum += pow((v.at(indexExcl).at(j) - v.at(i).at(j)), 2);
				}
			}
			dist = sqrt(sum);
		//	cout << "Dist: " << dist << endl;
			if (dist <= closest)
			{
				closest = dist;
				closeInd = i;
				//cout << "Curr closest: " << closest << endl;
			}
			sum = 0;
		}
	}
	
	//cout << "predicted val : " << v.at(closeInd).at(0) << endl;
	return v.at(closeInd).at(0);
	
}

float nearNeigh(vector<vector<float>> &v, vector<unsigned> &featureSub, bool allFlag)
{
	//currently for small set V
	//vector<unsigned> featureSub = {15, 27, 1};
	int classifiedAs = 1;
	int numCorrectPred = 0;
	
	//call nearest neighbor classifier on best features found
	for (unsigned k = 0; k < v.size(); ++k)
	{
		//classifiedAs = v.at(k).at(0);
		classifiedAs = hlprNN(featureSub, v, k, allFlag);
		//cout << "Classified as: " << classifiedAs << endl;
		int actual = v.at(k).at(0);
		//cout << "Actual: " << actual << endl;
		if (classifiedAs == actual)
		{
			++numCorrectPred;
			//cout << "Currently predicted right: " << numCorrectPred << endl;
		}
		//cout << endl << endl;
	}
	//cout << "Num correctly predicted : " << numCorrectPred << endl;
	//int instances = v.size();
	//cout << "Instances : " << instances << endl;
	//*/
	float accuracy = static_cast<float>(numCorrectPred) / static_cast<float>(v.size());
	//cout << "Accuracy: " << accuracy*100 << "%"<< endl;

	return accuracy*100;
}

vector<unsigned> forSel(vector<vector<float>> &v, float &bestAcc)
{
	vector<unsigned> tSet;
	vector<unsigned> fSet;
	vector<unsigned> cSet;
	float acc = 0;
	float currBAcc = 0;
	
	for (unsigned i = 1; i < v.at(0).size(); ++i)
	{
		for(unsigned j = 1; j < v.at(0).size(); ++j)
		{
			int jTmp = j;
			if (find(tSet.begin(), tSet.end(), jTmp) == tSet.end())
			{
				tSet.push_back(j);
				acc = nearNeigh(v, tSet, false);
				cout << "Using feature(s) {";
				display(tSet);
				cout << "} accuracy is " << acc << "%" << endl;
				if (acc > bestAcc)
				{
					bestAcc = acc;
					fSet = tSet;
				}
				if (acc > currBAcc)
				{
					currBAcc = acc;
					cSet = tSet;
				}
				tSet.pop_back();
			}
		}
		cout << endl;
		if (currBAcc < bestAcc)
		{
			cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
		}
		cout << "Feature set {";
		display(cSet);
		cout << "} was best, accuracy is " << currBAcc << "%" << endl << endl;
		tSet = cSet;
		cSet.clear();
		currBAcc = 0;
	}
	
	return fSet;
}

vector<unsigned> backSel(vector<vector<float>> &v, float &bestAcc)
{
	unsigned rng = v.at(0).size() - 2;
	unsigned rng2 = v.at(0).size() - 1;
	vector<unsigned> tSet (v.at(0).size());
	vector<unsigned> fSet;
	vector<unsigned> cSet;
	vector<unsigned>::iterator it; 
	unsigned val = 0;
	float acc = 0;
	float currBAcc = 0;
	//unsigned curr = 0;
	
	std::iota(tSet.begin(), tSet.end(), 0);
	tSet.erase(tSet.begin());
	//display(tSet);
	
	//cout << "Starting loop, rng2 is " << rng2 << endl;
	
	for (unsigned i = 0; i < rng; ++i)
	{
		for (unsigned j = 0; j < rng2; ++j)
		{
				//cout << "tSet before: " << endl;
				//display(tSet);
				val = tSet.at(j);
				tSet.erase(tSet.begin() + j);
				//cout << endl << "Eliminated : " << val << endl;
				//display(tSet);
				acc = nearNeigh(v, tSet, false);
				cout << "Using feature(s) {";
				display(tSet);
				cout << "} accuracy is " << acc << "%" << endl;
				if (acc > bestAcc)
				{
					bestAcc = acc;
					fSet = tSet;
				}
				if (acc > currBAcc)
				{
					currBAcc = acc;
					cSet = tSet;
				}
				tSet.insert(tSet.begin() + j, val);
			//}
			
		}
		
		cout << endl;
		if (currBAcc < bestAcc)
		{
			cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
		}
		cout << "Feature set {";
		display(cSet);
		cout << "} was best, accuracy is " << currBAcc << "%" << endl << endl;
		tSet = cSet;
		cSet.clear();
		currBAcc = 0;
		//cout << "LOOP" << endl;
		rng2 = rng2 - 1;
	}
	
	return fSet;
}

int main()
{
	string infile;
	int alg;
	float min, max, accuracy;
	vector<vector<float>> v;
	vector<unsigned> featureSub;
	bool allFlag = true;
	
	cout << "Welcome to Tia Moore's Feature Selection Algorithm." << endl;
	cout << "Type in the name of the file to test: ";
	cin >> infile;
	
	min = INFINITY;
	max = -1;
	
	bool dataLoaded = loadData(infile, v, min, max);
	if (dataLoaded == false)
	{
		cout << "Error loading data." << endl;
		return 0;
	}
	
	cout << "Type the number of the algorithm you want to run." << endl;
	cout << "	1) Forward Selection" << endl;
    cout << "	2) Backward Elimination" << endl;
    
    cin >> alg;
	
	cout << "This dataset has " << v.at(0).size() - 1 << " features (not including the class attribute), ";
    cout << "with " << v.size() << " instances." << endl << endl;
    cout << "Please wait while I normalize the data..." << endl;
    //min = v.at(0).at(1);
    //max = v.at(0).at(1);
    normalize(v, min, max);
	cout << "Done!" << endl << endl;

	accuracy = nearNeigh(v, featureSub, allFlag);
	cout << "Running nearest neighbor with all " << v.at(0).size() - 1 << " features,";
	cout << " using “leaving-one-out” evaluation, I get an accuracy of " << accuracy << "%" << endl << endl;
    
    allFlag = false;
    
    if (alg == 1)
    {
    	cout << "Beginning Search..." << endl << endl;
		featureSub = forSel(v, accuracy);
    }
    else if (alg == 2)
    {
    	cout << "Beginning Search..." << endl << endl;
		featureSub = backSel(v, accuracy);
    }
	else
	{
		cout << "Beginning Search..." << endl << endl;
		featureSub = forSel(v, accuracy);
	}
	
	//accuracy = nearNeigh(v, featureSub, allFlag);
	cout << "Finished search!! The best feature subset is {";
	display(featureSub);
	cout << "}, which has an accuracy of " << accuracy << "%" << endl;
    
	return 0;
}