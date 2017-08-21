#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "./libsvm/svm.h"
using namespace std;
//�ڴ����
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/************************************************************************/
/* ��װsvm                                                                     */
/************************************************************************/
class CxLibSVM
{
private:
	
	struct svm_model*	model_;
	struct svm_parameter	param;
	struct svm_problem		prob;
	struct svm_node *		x_space;
public:
	//************************************
	// ��    ��: ���캯��
	// ��    ��: CxLibSVM
	// �� �� ��: CxLibSVM::CxLibSVM
	// ����Ȩ��: public 
	// �� �� ֵ: 
	// �� �� ��:
	//************************************
	CxLibSVM()
	{
		model_ = NULL;
	}

	//************************************
	// ��    ��: ��������
	// ��    ��: ~CxLibSVM
	// �� �� ��: CxLibSVM::~CxLibSVM
	// ����Ȩ��: public 
	// �� �� ֵ: 
	// �� �� ��:
	//************************************
	~CxLibSVM()
	{
		free_model();
	}
	
	//************************************
	// ��    ��: ѵ��ģ��
	// ��    ��: train
	// �� �� ��: CxLibSVM::train
	// ����Ȩ��: public 
	// ��    ��: const vector<vector<double>> & x
	// ��    ��: const vector<double> & y
	// ��    ��: const int & alg_type
	// �� �� ֵ: void
	// �� �� ��:
	//************************************
	void train(const vector<vector<double>>&  x, const vector<double>& y)
	{
		if (x.size() == 0)return;
		//�ͷ���ǰ��ģ��
		free_model();
		/*��ʼ��*/		
		long	len = x.size();
		long	dim = x[0].size();
		long	elements = len*dim;

		//������ʼ�����������������������޸ļ���
		// Ĭ�ϲ���
		param.svm_type = C_SVC;		//�㷨����
		param.kernel_type = RBF;	//�˺�������
		param.degree = 3;	//����ʽ�˺����Ĳ���degree
		param.coef0 = 0;	//����ʽ�˺����Ĳ���coef0
		param.gamma = 0.1;	//1/num_features��rbf�˺�������
		param.nu = 0.5;		//nu-svc�Ĳ���
		param.C = 1;		//������ĳͷ�ϵ��
		param.eps = 1e-3;	//��������
		param.cache_size = 100;	//�����ڴ滺�� 100MB
		param.p = 0.1;	
		param.shrinking = 1;
		param.probability = 1;	//1��ʾѵ��ʱ���ɸ���ģ�ͣ�0��ʾѵ��ʱ�����ɸ���ģ�ͣ�����Ԥ���������������ĸ���
		param.nr_weight = 0;	//���Ȩ��
		param.weight = NULL;	//����Ȩ��
		param.weight_label = NULL;	//���Ȩ��

		//ת������Ϊlibsvm��ʽ
		prob.l = len;
		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(struct svm_node *, prob.l);
		x_space	= Malloc(struct svm_node, elements+len);
		int j = 0;
		for (int l = 0; l < len; l++)
		{
			prob.x[l] = &x_space[j];
			for (int d = 0; d < dim; d++)
			{				
				x_space[j].index = d+1;
				x_space[j].value = x[l][d];	
				j++;
			}
			x_space[j++].index = -1;
			prob.y[l] = y[l];
			//cout << y[l] << " ";
		}

		/*ѵ��*/
		model_ = svm_train(&prob, &param);
		//cout << model_ << endl;
	}

	//************************************
	// ��    ��: Ԥ����������������͸���
	// ��    ��: predict
	// �� �� ��: CxLibSVM::predict
	// ����Ȩ��: public 
	// ��    ��: const vector<double> & x	����
	// ��    ��: double & prob_est			�����Ƶĸ���
	// �� �� ֵ: double						Ԥ������
	// �� �� ��:
	//************************************
	int predict(const vector<double>& x,double& prob_est)
	{
		//����ת��
		svm_node* x_test = Malloc(struct svm_node, x.size()+1);
		for (unsigned int i=0; i<x.size(); i++)
		{
			x_test[i].index = i;
			x_test[i].value = x[i];
		}
		x_test[x.size()].index = -1;
		double *probs = new double[model_->nr_class];//�洢���������ĸ���
		//Ԥ�����͸���
		int value = (int)svm_predict_probability(model_, x_test, probs);
		//int value = (int)svm_predict(model_, x_test); //����svm_predict����Ԥ��  
		for (int k = 0; k < model_->nr_class;k++)
		{//����������Ӧ�ĸ���
			if (model_->label[k] == value)
			{
				prob_est = probs[k];
				break;
			}
		}
		delete[] probs;
		return value;
	}

	//************************************
	// ��    ��: ����svmģ��
	// ��    ��: load_model
	// �� �� ��: CxLibSVM::load_model
	// ����Ȩ��: public 
	// ��    ��: string model_path	ģ��·��
	// �� �� ֵ: int 0��ʾ�ɹ���-1��ʾʧ��
	// �� �� ��:
	//************************************
	int load_model(string model_path)
	{
		//�ͷ�ԭ����ģ��
		//free_model();
		//����ģ��
		model_ = svm_load_model(model_path.c_str());
		if (model_ == NULL)
			return -1;
		return 0;
	}

	//************************************
	// ��    ��: ����ģ��
	// ��    ��: save_model
	// �� �� ��: CxLibSVM::save_model
	// ����Ȩ��: public 
	// ��    ��: string model_path	ģ��·��
	// �� �� ֵ: int	0��ʾ�ɹ���-1��ʾʧ��
	// �� �� ��:
	//************************************
	int save_model(string model_path)
	{
		int flag = svm_save_model(model_path.c_str(), model_);
		return flag;
	}

private:

	//************************************
	// ��    ��: �ͷ�svmģ���ڴ�
	// ��    ��: free_model
	// �� �� ��: CxLibSVM::free_model
	// ����Ȩ��: private 
	// �� �� ֵ: void
	// �� �� ��:
	//************************************
	void free_model()
	{
		if (model_ != NULL)
		{
			svm_free_and_destroy_model(&model_);
			svm_destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
		}
	}
};