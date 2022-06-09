#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include "Utils.h"
#include <Python.h>


static int callPythonTrainElm(int num, const std::string& path)
{
	int res = 0;

	//Py_SetPythonHome(L"C:\\Program Files\\Python39");
	Py_Initialize();//使用python之前，要调用Py_Initialize();这个函数进行初始化
	if (!Py_IsInitialized())
	{
		printf("initializing fail！");
		system("pause");
		return 0;
	}
	
	PyRun_SimpleString("import sys");
	/*
	PyRun_SimpleString("import numpy as np");
	PyRun_SimpleString("d1d2uv = np.loadtxt('./data/inputnormalized.txt')");
	PyRun_SimpleString("print(d1d2uv.shape())");
	*/
	PyRun_SimpleString("print('Python initialized!')");
	PyRun_SimpleString("sys.path.append('./')");//这一步很重要，修改Python路径
	PyRun_SimpleString("print(sys.path)");

	PyObject * pModule = NULL;//声明变量
	PyObject * pFunc = NULL;// 声明变量

	pModule = PyImport_ImportModule("SolveElm");//这里是要调用的文件名hello.py
	if (pModule == NULL)
	{
		std::cout << "can not find 'SolveElm' File" << std::endl;
		system("pause");
		return 0;
	}

	pFunc = PyObject_GetAttrString(pModule, "trainElm");//这里是要调用的函数名
	if (pFunc == NULL)
	{
		std::cout << "can not find 'trainElm' function" << std::endl;
		system("pause");
		return 0;
	}
	PyObject* args = Py_BuildValue("(is)", num, path.c_str());//给python函数参数赋值

	printf("Build ELM model and Solve \n");
	PyObject* pRet = PyObject_CallObject(pFunc, args);//调用函数

	PyArg_Parse(pRet, "i", &res);//转换返回类型

	Py_Finalize();//调用Py_Finalize，这个根Py_Initialize相对应的。
	return res;
}

struct normalizationInfo
{
	cv::Mat max;
	cv::Mat min;
};

class ELM
{
public:
	ELM(std::string path);
	~ELM();

	void SetParam(int num, const cv::Mat& SampleInput, const cv::Mat& SampleOutput)
	{
		_NumOfNode = num;
		_SampleInput = SampleInput;
		_SampleOutput = SampleOutput;
	};
	void CompleteELM();
	void LoadELM();
	cv::Mat PredictLine(cv::Mat abuv, cv::Mat K, cv::Mat DistortPara);
	double PtoL(cv::Mat Point, cv::Mat Line);


	enum MapMinMax
	{
		APPLY = 0,
		REVERSE = 1
	};
	void mapminmax(const cv::Mat &input, cv::Mat &inputn, normalizationInfo &inputps);
	void mapminmax(const cv::Mat &input, cv::Mat &output, const normalizationInfo &inputps, int flag);
	void elmtrain(int NumOfNode, const std::string& path);
	cv::Mat elmpredict(const cv::Mat &input, const cv::Mat &IW, const cv::Mat &B, const cv::Mat &LW);
	void outputPara(const cv::Mat &IW, const  cv::Mat &B, const cv::Mat &LW, const std::string &dirpath);
	void outputStruct(const cv::Mat inputn, const cv::Mat outputn, const normalizationInfo &inputps, const normalizationInfo &outputps, const std::string &dirpath);

private:
	
	int _NumOfNode;

	cv::Mat _IW, _B, _LW;
	cv::Mat _SampleInput, _SampleOutput;
	cv::Mat _NormalizedInput, _NormalizedOutput;
	double _RMSE;
	normalizationInfo _inputps;
	normalizationInfo _outputps;

	std::string _lib_path;
	std::string _inputfilename, _outputfilename;
};

