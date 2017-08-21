#include "cxlibsvm.hpp"
#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
#include <sstream>

//ģ�庯������string���ͱ���ת��Ϊ���õ���ֵ���ͣ��˷��������ձ������ԣ�  
//template <class Type>
//Type stringToNum(const string& str)
//{
//	istringstream iss(str);
//	Type num;
//	iss >> num;
//	return num;
//}
void main()
{
	//��ʼ��libsvm
	CxLibSVM	svm;

	/*1��׼��ѵ������*/
	vector<vector<double>>	x;	//������
	vector<double>	y;			//�������
	int dim = 10;	//�������
	//��ȡѵ������
	ifstream in;
	in.open("train.txt");
	string feature;//ÿ������
	vector<double> lines; //�洢ÿ������
	vector<double> rx;
	double feat_onePoint;
	if (!in)
	{
		cout << "Error opening file"; exit(1);
	}
	while (!in.eof())
	{
		getline(in, feature);
		stringstream stringin(feature); //ʹ�ô���ʵ�ֶ�string�������������  
		lines.clear();
		while (stringin >> feat_onePoint) {      //���ո�һ�ζ�ȡһ�����ݴ���feat_onePoint   
			lines.push_back(feat_onePoint); //�洢ÿ�а��ո�ֿ�������
		}
		rx.clear();
		dim = lines.size();
		for (int j = 0; j < dim-1; j++)
		{
			rx.push_back(lines[j]);
		}
		x.push_back(rx);
		y.push_back(lines[dim-1]);
	}
	cout << "ѵ�����ݣ�" << x.size() << "��" << endl;

	////�����������������
	//for (int i = 0; i < sample_num; i++)
	//{
	//	vector<double> rx;
	//	for (int j = 0; j < dim; j++)
	//	{
	//		rx.push_back(scale*(rand() % 10) );
	//	}
	//	x.push_back(rx);
	//	y.push_back(1);
	//}

	////��������ĸ�������
	//for (int i = 0; i < sample_num; i++)
	//{
	//	vector<double> rx;
	//	for (int j = 0; j < dim; j++)
	//	{
	//		rx.push_back(-scale*(rand() % 10));
	//	}
	//	x.push_back(rx);
	//	y.push_back(2);
	//}

	/*2��ѵ��*/
	svm.train(x, y);

	/*3������ģ��*/
	string model_path = "svm_model.txt";
	svm.save_model(model_path);

	/*4������ģ��*/
	string model_path_p = "svm_model.txt";
	svm.load_model(model_path_p);

	/*5��Ԥ��*/
	//��ȡ��������
	ifstream in_test;
	in_test.open("test.txt");
	vector<double> x_test;
	int right_num = 0,num=0;
	if (!in_test)
	{
		cout << "Error opening file"; exit(1);
	}
	while (!in_test.eof())
	{
		getline(in_test, feature);
		stringstream stringin(feature); //ʹ�ô���ʵ�ֶ�string�������������  
		lines.clear();
		while (stringin >> feat_onePoint) {      //���ո�һ�ζ�ȡһ�����ݴ���feat_onePoint   
			lines.push_back(feat_onePoint); //�洢ÿ�а��ո�ֿ�������   
		}
		dim = lines.size();
		x_test.clear();
		for (int j = 0; j < dim-1; j++)
		{
			x_test.push_back(lines[j]);
		}
		double prob_est;
		//Ԥ��
		double value = svm.predict(x_test, prob_est);
		if ((int)value == (int)lines[dim - 1])
			right_num++;
		//��ӡԤ�����͸���
		printf("label:%f,prob:%f", value, prob_est);
		num++;
		cout << endl;
	}
	cout << "�������ݣ�" << num << "��" << endl;
	//cout << right_num << endl;
	double right;
	right = (double)right_num / (double)num;
	cout << "��ȷ�ʣ�" << right << endl;
	////���������������
	//vector<double> x_test;
	//for (int j = 0; j < dim; j++)
	//{
	//	x_test.push_back(scale*(rand() % 10));
	//}
	//double prob_est;
	////Ԥ��
	//double value = svm.predict(x_test, prob_est);

	////��ӡԤ�����͸���
	//printf("label:%f,prob:%f", value, prob_est);
}