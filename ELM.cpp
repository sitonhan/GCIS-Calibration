#include "ELM.h"



ELM::ELM(std::string lib_path)
{
	_lib_path = lib_path;
}


ELM::~ELM()
{
}

void ELM::CompleteELM()
{
	mapminmax(_SampleInput, _NormalizedInput, _inputps);
	mapminmax(_SampleOutput, _NormalizedOutput, _outputps);
	outputStruct(_NormalizedInput, _NormalizedOutput, _inputps, _outputps, _lib_path);
	elmtrain(_NumOfNode, _lib_path);
}

void ELM::LoadELM()
{
	std::string filenameIW = _lib_path + "/ModelParameter/IW.txt";
	std::string filenameB = _lib_path + "/ModelParameter/B.txt";
	std::string filenameLW = _lib_path + "/ModelParameter/LW.txt";
	std::string filename_inputs_min = _lib_path + "/ModelParameter/input_min.txt";
	std::string filename_inputs_max = _lib_path + "/ModelParameter/input_max.txt";
	std::string filename_outputs_min = _lib_path + "/ModelParameter/output_min.txt";
	std::string filename_outputs_max = _lib_path + "/ModelParameter/output_max.txt";

	Utils::ReadDataFromTxt(filenameIW, _IW);
	Utils::ReadDataFromTxt(filenameB, _B);
	Utils::ReadDataFromTxt(filenameLW, _LW);
	Utils::ReadDataFromTxt(filename_inputs_min, _inputps.min);
	Utils::ReadDataFromTxt(filename_inputs_max, _inputps.max);
	Utils::ReadDataFromTxt(filename_outputs_min, _outputps.min);
	Utils::ReadDataFromTxt(filename_outputs_max, _outputps.max);

	
}

cv::Mat ELM::PredictLine(cv::Mat abuv, cv::Mat K, cv::Mat DistortPara)
{
	//去畸变
	cv::Mat uv = abuv.rowRange(2, 4).clone();
	uv = uv.t();
	cv::Mat tem_uv = uv.reshape(2);
	cv::Mat undistort_uv;
	cv::undistortPoints(tem_uv, undistort_uv, K, DistortPara, cv::noArray(), K);
	cv::Mat tem_undistort_uv = undistort_uv.reshape(1);
	undistort_uv = undistort_uv.t();
	abuv.rowRange(2, 4) = undistort_uv;
	cv::Mat N_abuv;
	mapminmax(abuv, N_abuv, _inputps, APPLY);
	cv::Mat Tn_sim = elmpredict(N_abuv, _IW, _B, _LW);
	cv::Mat T_sim;
	mapminmax(Tn_sim, T_sim, _outputps, REVERSE);
	return T_sim.t();
}

double ELM::PtoL(cv::Mat Point, cv::Mat Line)
{
	double a1 = Line.at<double>(0) * (Point.at<double>(0) - Line.at<double>(3));
	double a2 = Line.at<double>(1) * (Point.at<double>(1) - Line.at<double>(4));
	double a3 = Line.at<double>(2) * (Point.at<double>(2) - Line.at<double>(5));
	double a = Line.at<double>(0) * Line.at<double>(0) + Line.at<double>(1) * Line.at<double>(1) + Line.at<double>(2) * Line.at<double>(2);
	double t = (a1 + a2 + a3) / a;
	cv::Mat point1 = cv::Mat::zeros(3, 1, CV_64F);
	cv::Mat point2 = cv::Mat::zeros(3, 1, CV_64F);
	point1.at<double>(0) = Line.at<double>(0) * t + Line.at<double>(3);
	point1.at<double>(1) = Line.at<double>(1) * t + Line.at<double>(4);
	point1.at<double>(2) = Line.at<double>(2) * t + Line.at<double>(5);
	Point.copyTo(point2);
	if (point1.rows != point2.rows)
	{
		point1 = point1.t();
	}
	cv::Point3d error = cv::Point3d(point1.at<double>(0) - point2.at<double>(0),
		point1.at<double>(1) - point2.at<double>(1),
		point1.at<double>(2) - point2.at<double>(2));
	double distance = cv::norm(error);
	return distance;
}

void ELM::mapminmax(const cv::Mat &input, cv::Mat &inputn, normalizationInfo &inputps)
{
	int col = input.cols; int row = input.rows;
	cv::Mat bmax = cv::Mat::zeros(row, 1, CV_64F);
	cv::Mat bmin = cv::Mat::zeros(row, 1, CV_64F);
	cv::reduce(input, bmax, 0, cv::REDUCE_MAX);
	cv::reduce(input, bmin, 0, cv::REDUCE_MIN);
	bmax.copyTo(inputps.max);
	bmax = bmax.t();
	bmax.copyTo(inputps.max);
	bmin = bmin.t();
	bmin.copyTo(inputps.min);
	inputn = cv::Mat::zeros(row, col, CV_64F);
	for (int i = 0; i < col; ++i)
	{
		if (inputps.min.at<double>(i) == inputps.max.at<double>(i))
		{
			cv::Mat Col = cv::Mat::ones(row, 1, CV_64F);
			Col.copyTo(inputn.col(i));
		}
		else
		{
			cv::Mat Xmin = cv::Mat::ones(row, 1, CV_64F)*inputps.min.at<double>(i);
			cv::Mat Xmax = cv::Mat::ones(row, 1, CV_64F)*inputps.max.at<double>(i);
			cv::Mat Ymin = cv::Mat::ones(row, 1, CV_64F)*(-1);
			cv::Mat Col = 2 * (input.col(i) - Xmin) / (Xmax - Xmin) + Ymin;
			Col.copyTo(inputn.col(i));
		}
	}
}

void ELM::mapminmax(const cv::Mat &input, cv::Mat &output, const normalizationInfo &inputps, int flag)
{
	int col = input.cols;
	int row = input.rows;
	output = cv::Mat::zeros(row, col, CV_64F);
	if (row == inputps.max.rows)
	{
		if (flag == MapMinMax::APPLY)
		{
			for (int i = 0; i < row; ++i)
			{
				if (inputps.min.at<double>(i) == inputps.max.at<double>(i))
				{
					cv::Mat Row = cv::Mat::ones(1, col, CV_64F);
					Row.copyTo(output.row(i));
				} 
				else
				{
					cv::Mat Xmin = cv::Mat::ones(1, col, CV_64F) * inputps.min.at<double>(i);
					cv::Mat Xmax = cv::Mat::ones(1, col, CV_64F) * inputps.max.at<double>(i);
					cv::Mat Ymin = cv::Mat::ones(1, col, CV_64F) * (-1);
					cv::Mat Row = 2 * (input.row(i) - Xmin) / (Xmax - Xmin) + Ymin;	//归一化到（-1，1）区间
					Row.copyTo(output.row(i));
				}
			}
		}
		else if (flag == MapMinMax::REVERSE)
		{
			for (int i = 0; i < row; ++i)
			{
				if (inputps.min.at<double>(i) == inputps.max.at<double>(i))
				{
					cv::Mat Row = cv::Mat::ones(1, col, CV_64F) * inputps.min.at<double>(i);
					Row.copyTo(output.row(i));
				} 
				else
				{
					cv::Mat Xmin = cv::Mat::ones(1, col, CV_64F) * inputps.min.at<double>(i);
					cv::Mat Xmax = cv::Mat::ones(1, col, CV_64F) * inputps.max.at<double>(i);
					double max = inputps.max.at<double>(i);
					double min = inputps.min.at<double>(i);
					cv::Mat Ymin = cv::Mat::ones(1, col, CV_64F) * (-1);
					cv::Mat Row = 0.5 * (input.row(i) - Ymin) * (max - min) + Xmin;
					Row.copyTo(output.row(i));
				}
				
			}
		}
	}
	else if (col == inputps.max.rows)
	{
		cv::Mat input_t = input.t();
		if (flag == MapMinMax::APPLY)
		{
			for (int i = 0; i < row; ++i)
			{
				if (inputps.min.at<double>(i) == inputps.max.at<double>(i))
				{
					cv::Mat Row = cv::Mat::ones(1, col, CV_64F);
					Row.copyTo(output.row(i));
				}
				else
				{
					cv::Mat Xmin = cv::Mat::ones(1, col, CV_64F) * inputps.min.at<double>(i);
					cv::Mat Xmax = cv::Mat::ones(1, col, CV_64F) * inputps.max.at<double>(i);
					cv::Mat Ymin = cv::Mat::ones(1, col, CV_64F) * (-1);
					cv::Mat Row = 2 * (input.row(i) - Xmin) / (Xmax - Xmin) + Ymin;	//归一化到（-1，1）区间
					Row.copyTo(output.row(i));
				}
			}
		}
		else if (flag == MapMinMax::REVERSE)
		{
			for (int i = 0; i < row; ++i)
			{
				if (inputps.min.at<double>(i) == inputps.max.at<double>(i))
				{
					cv::Mat Row = cv::Mat::ones(1, col, CV_64F) * inputps.min.at<double>(i);
					Row.copyTo(output.row(i));
				}
				else
				{
					cv::Mat Xmin = cv::Mat::ones(1, col, CV_64F) * inputps.min.at<double>(i);
					cv::Mat Xmax = cv::Mat::ones(1, col, CV_64F) * inputps.max.at<double>(i);
					double max = inputps.max.at<double>(i);
					double min = inputps.min.at<double>(i);
					cv::Mat Ymin = cv::Mat::ones(1, col, CV_64F) * (-1);
					cv::Mat Row = 0.5 * (input.row(i) - Ymin) * (max - min) + Xmin;
					Row.copyTo(output.row(i));
				}

			}
		}
	}
	else
	{
		std::cout << "mapmaxmin size error!" << std::endl;
		CV_Assert(1);
	}
}

void ELM::elmtrain(int NumOfNode, const std::string& path)
{
	
	int res = callPythonTrainElm(NumOfNode, path);
	if (res == 0)
	{
		printf("Elm solve Error! \n");
		system("pause");
	}
}

cv::Mat ELM::elmpredict(const cv::Mat &input, const cv::Mat &IW, const cv::Mat &B, const cv::Mat &LW)
{
	int Q = input.cols;
	cv::Mat BiasMatrix = repeat(B, 1, Q);
	cv::Mat tempH = IW * input + BiasMatrix; 
	cv::Mat H = cv::Mat::zeros(tempH.rows, tempH.cols, CV_64F);
	for (int i = 0; i < tempH.rows; ++i)
	{
		for (int j = 0; j < tempH.cols; ++j)
		{
			H.at<double>(i, j) = 1 / (exp(-tempH.at<double>(i, j)) + 1);
		}
	}
	cv::Mat output = H.t()*LW;
	return output.t();
}

void ELM::outputPara(const cv::Mat &IW, const cv::Mat &B, const cv::Mat &LW, const std::string &dirpath)
{
	int N = B.rows;
	std::ofstream outIW;
	std::string filepath_IW = dirpath + "/ModelParameter/IW.txt";
	outIW.open(filepath_IW);
	outIW << std::setprecision(18);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < IW.cols; ++j)
		{
			outIW << IW.at<double>(i, j)<<" ";
		}
		outIW << std::endl;
	}

	std::ofstream outB;
	std::string filepath_B = dirpath + "/ModelParameter/B.txt";
	outB.open(filepath_B);
	outB << std::setprecision(18);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < B.cols; ++j)
		{
			outB << B.at<double>(i, j) << " ";
		}
		outB << std::endl;
	}
	std::ofstream outLW;
	std::string filepath_LW = dirpath + "/ModelParameter/LW.txt";
	outLW.open(filepath_LW);
	outLW << std::setprecision(18);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < LW.cols; ++j)
		{
			outLW << LW.at<double>(i, j) << " ";
		}
		outLW << std::endl;
	}
}

void ELM::outputStruct(const cv::Mat inputn, const cv::Mat outputn, const normalizationInfo &inputps, const normalizationInfo &outputps, const std::string &dirpath)
{
	Utils::WriteDataToTxt(inputn, dirpath + "/inputnormalized.txt");
	Utils::WriteDataToTxt(outputn, dirpath + "/outputnormalized.txt");
	{
		std::ofstream out_inMin;
		std::string inMin = dirpath + "/ModelParameter/input_min.txt";
		out_inMin.open(inMin);
		out_inMin << std::setprecision(18);
		for (int i = 0; i < inputps.min.rows; ++i)
		{
			out_inMin << inputps.min.at<double>(i, 0) << std::endl;
		}
		out_inMin.close();
	}
	{
		std::ofstream out_inMax;
		std::string inMax = dirpath + "/ModelParameter/input_max.txt";
		out_inMax.open(inMax);
		out_inMax << std::setprecision(18);
		for (int i = 0; i < inputps.max.rows; ++i)
		{
			out_inMax << inputps.max.at<double>(i, 0) << std::endl;
		}
		out_inMax.close();
	}
	{
		std::ofstream out_outMin;
		std::string outMin = dirpath + "/ModelParameter/output_min.txt";
		out_outMin.open(outMin);
		out_outMin << std::setprecision(18);
		for (int i = 0; i < outputps.min.rows; ++i)
		{
			out_outMin << outputps.min.at<double>(i, 0) << std::endl;
		}
	}
	{
		std::ofstream out_outMax;
		std::string outMax = dirpath + "/ModelParameter/output_max.txt";
		out_outMax.open(outMax);
		out_outMax << std::setprecision(18);
		for (int i = 0; i < outputps.max.rows; ++i)
		{
			out_outMax << outputps.max.at<double>(i, 0) << std::endl;
		}
	}
}



