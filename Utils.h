#pragma once

#include <unordered_map>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <io.h>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "tinyxml2.h"

namespace Utils
{
	static std::string doublevec2string(const std::vector<double> src);

	static std::vector<double> string2doublevec(const std::string str);

	static std::vector<std::string> split(const std::string & str, const std::string & pattern);

	static void ReadDataFromTxt(std::string path, cv::Mat& data);

	static void WriteDataToTxt(std::vector<cv::Point3d>& data, std::string path);

	static void WriteDataToTxt(std::vector<cv::Point2d>& data, std::string path);

	static void WriteDataToTxt(cv::Mat data, std::string path);

	static void eigen2cv(const Eigen::Vector3d& eigen, cv::Point3d& cv)
	{
		cv = cv::Point3d(eigen[0], eigen[1], eigen[2]);
	}
	
	static void cv2eigen(const cv::Point3d& cv, Eigen::Vector3d& eigen)
	{
		eigen = Eigen::Vector3d(cv.x, cv.y, cv.z);
	}

	static void calculateStddev(const std::vector<double>& deviationsX,
		double& max, double& mean, double& stddev)
	{
		int n = deviationsX.size();
		auto iter = std::max_element(deviationsX.begin(), deviationsX.end());
		max = *iter;
		mean = std::accumulate(deviationsX.begin(), deviationsX.end(), 0.) / n;
		double stddevx = 0.;
		for (auto& deviationX : deviationsX) {
			stddevx += ((deviationX - mean) * (deviationX - mean));
		}
		stddev = std::sqrt(stddevx / (n - 1));
	}


}

std::string Utils::doublevec2string(const std::vector<double> src)
{
	std::string str;
	for (auto d : src)
	{
		str.append(std::to_string(d));
		str.append(" ");
	}
	return str;
}

std::vector<double> Utils::string2doublevec(const std::string str)
{
	std::vector<std::string> v_str = Utils::split(str, " ");
	std::vector<double> temp;
	for (int i = 0; i < v_str.size(); ++i)
	{
		if (v_str[i].size())
		{
			temp.emplace_back(std::stod(v_str[i]));
		}
	}
	return temp;
}

std::vector<std::string> Utils::split(const std::string & str, const std::string & pattern)
{
	std::vector<std::string> res;
	if (str == "")
		return res;
	//在字符串末尾也加入分隔符，方便截取最后一段
	std::string strs = str + pattern;
	size_t pos = strs.find(pattern);

	while (pos != strs.npos)
	{
		std::string temp = strs.substr(0, pos);
		res.push_back(temp);
		//去掉已分割的字符串,在剩下的字符串中进行分割
		strs = strs.substr(pos + 1, strs.size());
		pos = strs.find(pattern);
	}

	return res;
}

void Utils::ReadDataFromTxt(std::string path, cv::Mat& data)
{
	data.release();
	std::ifstream infile;
	infile.open(path, std::ios::in);
	if (infile.is_open())
	{
		while (!infile.eof())
		{
			std::string str_line;
			getline(infile, str_line);
			std::vector<std::string> line_vec = Utils::split(str_line, " ");
			if (line_vec.size())
			{
				std::vector<double> line;
				for (int i = 0; i < line_vec.size(); i++)
				{
					if (line_vec[i].size())
					{
						line.push_back(std::stod(line_vec[i]));
					}
				}
				cv::Mat tem_line(line);
				data.push_back(tem_line.t());
			}
		}
	}
	else
	{
		std::cout << path << " is not found!" << std::endl;
	}
	infile.close();
}

void Utils::WriteDataToTxt(std::vector<cv::Point3d>& data, std::string path)
{
	std::ofstream outfile;
	outfile.open(path, std::ios::out);
	outfile << std::setprecision(10);
	for (auto item : data)
	{
		outfile << item.x << " " << item.y << " " << item.z << std::endl;
	}
	outfile.close();
}

void Utils::WriteDataToTxt(std::vector<cv::Point2d>& data, std::string path)
{
	std::ofstream outfile;
	outfile.open(path, std::ios::out);
	outfile << std::setprecision(10);

	for (auto item : data)
	{
		outfile << item.x << " " << item.y << std::endl;
	}
	outfile.close();
}

void Utils::WriteDataToTxt(cv::Mat data, std::string path)
{
	std::ofstream outfile;
	outfile.open(path, std::ios::out);
	outfile << std::setprecision(10);

	for (size_t i = 0; i < data.rows; i++)
	{
		for (size_t j = 0; j < data.cols; j++)
		{
			outfile << data.at<double>(i, j) << " ";
		}
		outfile << std::endl;
	}
	outfile.close();
}