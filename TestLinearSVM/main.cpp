#include "cxlibsvm.hpp"
#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
#include <sstream>

//模板函数：将string类型变量转换为常用的数值类型（此方法具有普遍适用性）  
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
	//初始化libsvm
	CxLibSVM	svm;

	/*1、准备训练数据*/
	vector<vector<double>>	x;	//样本集
	vector<double>	y;			//样本类别集
	int dim = 10;	//样本类别
	//读取训练样本
	ifstream in;
	in.open("train.txt");
	string feature;//每行数据
	vector<double> lines; //存储每行数据
	vector<double> rx;
	double feat_onePoint;
	if (!in)
	{
		cout << "Error opening file"; exit(1);
	}
	while (!in.eof())
	{
		getline(in, feature);
		stringstream stringin(feature); //使用串流实现对string的输入输出操作  
		lines.clear();
		while (stringin >> feat_onePoint) {      //按空格一次读取一个数据存入feat_onePoint   
			lines.push_back(feat_onePoint); //存储每行按空格分开的数据
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
	cout << "训练数据：" << x.size() << "条" << endl;

	////生成随机的正类样本
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

	////生成随机的负类样本
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

	/*2、训练*/
	svm.train(x, y);

	/*3、保存模型*/
	string model_path = "svm_model.txt";
	svm.save_model(model_path);

	/*4、导入模型*/
	string model_path_p = "svm_model.txt";
	svm.load_model(model_path_p);

	/*5、预测*/
	//读取测试数据
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
		stringstream stringin(feature); //使用串流实现对string的输入输出操作  
		lines.clear();
		while (stringin >> feat_onePoint) {      //按空格一次读取一个数据存入feat_onePoint   
			lines.push_back(feat_onePoint); //存储每行按空格分开的数据   
		}
		dim = lines.size();
		x_test.clear();
		for (int j = 0; j < dim-1; j++)
		{
			x_test.push_back(lines[j]);
		}
		double prob_est;
		//预测
		double value = svm.predict(x_test, prob_est);
		if ((int)value == (int)lines[dim - 1])
			right_num++;
		//打印预测类别和概率
		printf("label:%f,prob:%f", value, prob_est);
		num++;
		cout << endl;
	}
	cout << "测试数据：" << num << "条" << endl;
	//cout << right_num << endl;
	double right;
	right = (double)right_num / (double)num;
	cout << "正确率：" << right << endl;
	////生成随机测试数据
	//vector<double> x_test;
	//for (int j = 0; j < dim; j++)
	//{
	//	x_test.push_back(scale*(rand() % 10));
	//}
	//double prob_est;
	////预测
	//double value = svm.predict(x_test, prob_est);

	////打印预测类别和概率
	//printf("label:%f,prob:%f", value, prob_est);
}