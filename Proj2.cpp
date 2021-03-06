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
    
    //uses filestream to open and read file
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
	 		//computes min and max values in file for normalization later
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
	 
    fin.close();
    
    return true;
}

void normalize(vector<vector<float>> &v, float &min, float &max)
{
	float tmp;
	
	//normalizes by rescaling
	float denom = max - min;
	for (unsigned i = 0; i < v.size(); ++i)
	{
		for (unsigned j = 1; j < v.at(i).size(); ++j)
		{
			tmp = (v.at(i).at(j) - min) / denom;
			v.at(i).at(j) = tmp;   
		}
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
}

bool isInFeatSub(unsigned &num, vector<unsigned> &featureSub)
{
	//used to know whether to use the feature when computing sum
	//during leave-one-out validator 
	for (unsigned i = 0; i < featureSub.size(); ++i)
	{
		if (num == featureSub.at(i))
		{
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
		if (i != indexExcl)
		{
		
			for (unsigned j = 1; j < v.at(i).size(); ++j)
			{
				if (isInFeatSub(j, featureSub) || allFlag == true)
				{
					//computes sum of each feature in set of each instance
					//distance away from current leave-one-out
					sum += pow((v.at(indexExcl).at(j) - v.at(i).at(j)), 2);
				}
			}
			dist = sqrt(sum);
			//finds distance from current leave-one-out to currently viewed instance
			if (dist <= closest)
			{
				closest = dist;
				closeInd = i;
			}
			sum = 0;
		}
	}
	
	return v.at(closeInd).at(0); //returns closest found instance
	
}

float nearNeigh(vector<vector<float>> &v, vector<unsigned> &featureSub, bool allFlag)
{
	int classifiedAs = 1;
	int numCorrectPred = 0;
	
	//call nearest neighbor classifier on best features found
	for (unsigned k = 0; k < v.size(); ++k)
	{
		//uses helper to determine classification 
		classifiedAs = hlprNN(featureSub, v, k, allFlag);
		int actual = v.at(k).at(0);
		if (classifiedAs == actual) //if correctly classified
		{
			++numCorrectPred;
		}
	}

	float accuracy = static_cast<float>(numCorrectPred) / static_cast<float>(v.size());

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
			//if the feature is not already in the set
			int jTmp = j;
			if (find(tSet.begin(), tSet.end(), jTmp) == tSet.end())
			{
				//adds new feature to the set
				tSet.push_back(j);
				//tests its accuracy with new feature
				acc = nearNeigh(v, tSet, false);
				cout << "Using feature(s) {";
				display(tSet);
				cout << "} accuracy is " << acc << "%" << endl;
				if (acc > bestAcc) //sets new overall best acc if its better
				{
					bestAcc = acc;
					fSet = tSet;
				}
				if (acc > currBAcc) //sets local maxima found
				{
					currBAcc = acc;
					cSet = tSet;
				}
				tSet.pop_back(); //takes off temp added feature
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
		//resets for next instance
		tSet = cSet;
		cSet.clear();
		currBAcc = 0;
	}
	
	return fSet; //returns best found feature set
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
	
	//sets tSet initially with features 1-n
	std::iota(tSet.begin(), tSet.end(), 0);
	tSet.erase(tSet.begin());

	
	for (unsigned i = 0; i < rng; ++i)
	{
		for (unsigned j = 0; j < rng2; ++j)
		{
				//erases one of the features to test if its
				//elimination is ideal
				val = tSet.at(j);
				tSet.erase(tSet.begin() + j);
				
				//finds acc after removing 1 feature
				acc = nearNeigh(v, tSet, false);
				cout << "Using feature(s) {";
				display(tSet);
				cout << "} accuracy is " << acc << "%" << endl;
				if (acc > bestAcc) //set best overall accuracy found if its better
				{
					bestAcc = acc;
					fSet = tSet;
				}
				if (acc > currBAcc) //sets the local maxima found
				{
					currBAcc = acc;
					cSet = tSet;
				}
				tSet.insert(tSet.begin() + j, val); //adds the eliminated val back
			
		}
		
		cout << endl;
		if (currBAcc < bestAcc) //if curr new accuracy goes down
		{
			cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
		}
		cout << "Feature set {";
		display(cSet);
		cout << "} was best, accuracy is " << currBAcc << "%" << endl << endl;
		//resets temp to the currbest set and clears currbest
		tSet = cSet;
		cSet.clear();
		currBAcc = 0;
		rng2 = rng2 - 1; //range should get smaller each go around
	}
	
	return fSet; //returns best found feature set
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
	
	//loads from the file and finds min,max for normalization
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
    normalize(v, min, max);
	cout << "Done!" << endl << endl;

	//calls nearest neighbor using all features for comparison
	accuracy = nearNeigh(v, featureSub, allFlag);
	cout << "Running nearest neighbor with all " << v.at(0).size() - 1 << " features,";
	cout << " using “leaving-one-out” evaluation, I get an accuracy of " << accuracy << "%" << endl << endl;
    
    allFlag = false;
    
    //determines alg to run and calls it
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
	
	cout << "Finished search!! The best feature subset is {";
	display(featureSub);
	cout << "}, which has an accuracy of " << accuracy << "%" << endl;
    
	return 0;
}