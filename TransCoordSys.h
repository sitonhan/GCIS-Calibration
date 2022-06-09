#pragma once
#pragma once

#include <unordered_map>
#include <filesystem>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "Utils.h"

namespace TransSys
{

	enum class FITMODE
	{
		LinerSolve,
		CeresSolve
	};

	class FitPlane
	{

	public:
		FitPlane(const std::vector<cv::Point3d>& pts, FITMODE fitmode);

		~FitPlane();
		void coarseFitting();
		void fineFitting();
		void setCoarseCoeff(const std::vector<double>& planeCoeff);
		std::vector<double> getPlaneCoeff();

	private:
		FITMODE _fitMode;
		std::vector<cv::Point3d> _pts;
		std::vector<double> _initialCoeff;
		std::vector<double> _finalCoeff;
	};

	inline FitPlane::FitPlane(const std::vector<cv::Point3d>& pts, TransSys::FITMODE fitmode)
		:_pts(pts), _fitMode(fitmode)
	{
		switch (_fitMode)
		{
		case FITMODE::LinerSolve:
			coarseFitting();
			break;
		case FITMODE::CeresSolve:
			coarseFitting();
			fineFitting();
			break;
		default:
			break;
		}
	}

	inline FitPlane::~FitPlane()
	{
	}

	inline void FitPlane::coarseFitting()
	{
		cv::Mat dst = cv::Mat(3, 3, CV_64F, cv::Scalar(0));//初始化系数矩阵A
		cv::Mat out = cv::Mat(3, 1, CV_64F, cv::Scalar(0));//初始化矩阵b
		for (int i = 0; i < _pts.size(); i++)
		{
			//计算3*3的系数矩阵
			dst.at<double>(0, 0) = dst.at<double>(0, 0) + pow(_pts[i].x, 2);
			dst.at<double>(0, 1) = dst.at<double>(0, 1) + _pts[i].x*_pts[i].y;
			dst.at<double>(0, 2) = dst.at<double>(0, 2) + _pts[i].x;
			dst.at<double>(1, 0) = dst.at<double>(1, 0) + _pts[i].x*_pts[i].y;
			dst.at<double>(1, 1) = dst.at<double>(1, 1) + pow(_pts[i].y, 2);
			dst.at<double>(1, 2) = dst.at<double>(1, 2) + _pts[i].y;
			dst.at<double>(2, 0) = dst.at<double>(2, 0) + _pts[i].x;
			dst.at<double>(2, 1) = dst.at<double>(2, 1) + _pts[i].y;
			dst.at<double>(2, 2) = _pts.size();
			//计算3*1的结果矩阵
			out.at<double>(0, 0) = out.at<double>(0, 0) + _pts[i].x*_pts[i].z;
			out.at<double>(1, 0) = out.at<double>(1, 0) + _pts[i].y*_pts[i].z;
			out.at<double>(2, 0) = out.at<double>(2, 0) + _pts[i].z;
		}
		//判断矩阵是否奇异
		double determ = determinant(dst);
		if (abs(determ) < 0.001) {
			std::cout << "矩阵奇异" << std::endl;
			return;
		}
		cv::Mat inv;
		cv::invert(dst, inv, cv::DECOMP_SVD);//求矩阵的逆
		cv::Mat output = inv * out;//计算输出
		_initialCoeff.clear();//把结果输出
		_initialCoeff.push_back(output.at<double>(0, 0));
		_initialCoeff.push_back(output.at<double>(1, 0));
		_initialCoeff.push_back(1);
		_initialCoeff.push_back(output.at<double>(2, 0));
	}

	struct CostFunctor
	{
		CostFunctor(double x, double y, double z) :_x(x), _y(y), _z(z) {}
		//距离d^2 = (ax+by+cz+d)^2/(a^2+b^2+c^2)
		template <typename T>
		bool operator()(const T* const a, const T* const b, const T* const c, const T* const d, T* residual) const
		{
			T temp = a[0] * _x + b[0] * _y + c[0] * _z + d[0];
			residual[0] = temp * temp / (a[0] * a[0] + b[0] * b[0] + c[0] * c[0]);
			return true;
		}

		double _x, _y, _z;
	};

	inline void FitPlane::fineFitting()
	{
		// 寻优参数的初值
		double initial_a = _initialCoeff[0];
		double initial_b = _initialCoeff[1];
		double initial_c = _initialCoeff[2];
		double initial_d = _initialCoeff[3];

		// 第二部分：构建寻优问题
		ceres::Problem problem;
		for (int i = 0; i < _pts.size(); i++)
		{
			//使用自动求导，将之前的代价函数结构体传入，第一个1是输出维度，即残差的维度，后4个1是输入维度，即待寻优参数的维度。
			ceres::CostFunction* cost_function =
				new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1, 1, 1>(new CostFunctor(_pts[i].x, _pts[i].y, _pts[i].z));
			//向问题中添加误差项，本问题比较简单，添加一个就行。
			problem.AddResidualBlock(cost_function, nullptr, &initial_a, &initial_b, &initial_c, &initial_d);
		}

		//第三部分： 配置并运行求解器
		ceres::Solver::Options options;
		//配置增量方程的解法
		options.linear_solver_type = ceres::DENSE_QR;
		//输出到cout
		options.minimizer_progress_to_stdout = true;
		//优化信息
		ceres::Solver::Summary summary;
		Solve(options, &problem, &summary);//求解!!!

		std::cout << summary.BriefReport() << "\n";//输出优化的简要信息
	  //最终结果
		std::vector<double> tempFinalCoeff = { initial_a ,initial_b ,initial_c ,initial_d };
		_finalCoeff = tempFinalCoeff;
	}

	inline void FitPlane::setCoarseCoeff(const std::vector<double>& planeCoeff)
	{
		_initialCoeff = planeCoeff;
	}

	inline std::vector<double> FitPlane::getPlaneCoeff()
	{
		switch (_fitMode)
		{
		case FITMODE::LinerSolve:
			return _initialCoeff;
		case FITMODE::CeresSolve:
			return _finalCoeff;
		}
	}



	static bool FitPlaneByPoints(const std::vector<Eigen::Vector3d>& points,
		double& a, double& b, double& c, double& d)
	{
		if (points.size() < 3) return false;

		double X, Y, Z;
		double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
		double sumSquareX = 0.0, sumSquareY = 0.0, sumSquareZ = 0.0;
		double sumXY = 0.0, sumXZ = 0.0, sumYZ = 0.0;
		for (const auto& p : points) {
			X = p(0);
			Y = p(1);
			Z = p(2);

			sumX += X;
			sumY += Y;
			sumZ += Z;

			sumSquareX += X * X;
			sumSquareY += Y * Y;
			sumSquareZ += Z * Z;

			sumXY += X * Y;
			sumXZ += X * Z;
			sumYZ += Y * Z;
		}

		Eigen::Matrix3d A;
		Eigen::Vector3d B;
		double n = (double)points.size();
		A << sumSquareX, sumXY, sumX,
			sumXY, sumSquareY, sumY,
			sumX, sumY, n;
		B << sumXZ, sumYZ, sumZ;

		//判断是否奇异
		double determ = A.determinant();
		if (abs(determ) < 0.001) {
			std::cout << "矩阵奇异" << std::endl;
			return false;
		}
		// A*x=B,最小二乘解法：x= inv(A'A)*A'b
		Eigen::Matrix3d AtA;
		Eigen::Matrix3d invAtA;
		Eigen::Vector3d AtB;
		Eigen::Vector3d solve;
		AtA = A.transpose() * A;
		invAtA = AtA.inverse();
		AtB = A.transpose() * B;
		solve = invAtA * AtB;
		Eigen::Vector3d coeffPlane = A.inverse()*B;
		a = coeffPlane(0);
		b = coeffPlane(1);
		c = 1.0;
		d = coeffPlane(2);

		return true;
	}

	static bool CalLinePlaneIntersecPoint(const double& a, const double& b, const double& c, const double& d,
		const Eigen::Vector3d& linePoint, const Eigen::Vector3d& lineDirection, Eigen::Vector3d& intersecPoint)
	{
		if (c > -1e-6 && c < 1e-6) return false;// 平面退化为直线

		Eigen::Vector3d planePoint(0, 0, -d / c);
		Eigen::Vector3d planeDirection(a, b, c);
		double tmp1 = (planePoint - linePoint).dot(planeDirection);
		double tmp2 = lineDirection.dot(planeDirection);

		if (tmp2 > -1e-6 && tmp2 < 1e-16) return false;// 直线与平面平行

		double lambda = tmp1 / tmp2;
		intersecPoint = linePoint + lambda * lineDirection;

		return true;
	}

	//待变换点
	//x-y平面点
	//近似原点
	//x轴正方向
	//转换矩阵
	static bool CoorTransform(std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& plane_points
		, Eigen::Vector3d& o_point, Eigen::Vector3d& x_point, Eigen::Isometry3d& T)
	{
		// 拟合平面
		double a, b, c, d;
		if (!FitPlaneByPoints(plane_points, a, b, c, d))
			return false;
		Eigen::Vector3d newZaxis(a, b, c);// 新的Z轴

		Eigen::Vector3d origin, newXaxis, temp1;
		if (!CalLinePlaneIntersecPoint(a, b, c, d, o_point, newZaxis, origin))
			return false;
		if (!CalLinePlaneIntersecPoint(a, b, c, d, x_point, newZaxis, temp1))
			return false;
		newXaxis = temp1 - origin;// 新的X轴

		Eigen::Vector3d newYaxis = newZaxis.cross(newXaxis);// 新的Y轴

		// 归一化
		newXaxis.normalize();
		newYaxis.normalize();
		newZaxis.normalize();

		// 构造变换矩阵
		Eigen::Matrix3d R;
		R.col(0) = newXaxis;
		R.col(1) = newYaxis;
		R.col(2) = newZaxis;

		T = Eigen::Isometry3d::Identity();
		T.rotate(R);
		T.pretranslate(origin);
		//std::cout << T.matrix() << std::endl;

		T = T.inverse();
		//std::cout << T.matrix() << std::endl;

		// 变换点坐标
		for (auto& it : points)
			it = T * it;

		return true;
	}

	static bool CoorTransform(const cv::Vec4d& plane, const cv::Point3d& o, const cv::Point3d& x, Eigen::Isometry3d& T)
	{
		// 拟合平面
		double a, b, c, d;
		a = plane[0];
		b = plane[1];
		c = plane[2];
		d = plane[3];
		Eigen::Vector3d o_point, origin, x_point, newXaxis, temp1;
		o_point = Eigen::Vector3d(o.x, o.y, o.z);
		Eigen::Vector3d newZaxis(a, b, c);// 新的Z轴
		if (!CalLinePlaneIntersecPoint(a, b, c, d, o_point, newZaxis, origin))
			return false;
		x_point = Eigen::Vector3d(x.x, x.y, x.z);
		if (!CalLinePlaneIntersecPoint(a, b, c, d, x_point, newZaxis, temp1))
			return false;
		newXaxis = temp1 - origin;// 新的X轴

		Eigen::Vector3d newYaxis = newZaxis.cross(newXaxis);// 新的Y轴

		// 归一化
		newXaxis.normalize();
		newYaxis.normalize();
		newZaxis.normalize();

		// 构造变换矩阵
		Eigen::Matrix3d R;
		R.col(0) = newXaxis;
		R.col(1) = newYaxis;
		R.col(2) = newZaxis;

		T = Eigen::Isometry3d::Identity();
		T.rotate(R);
		T.pretranslate(origin);
		//std::cout << T.matrix() << std::endl;

		T = T.inverse();
		//std::cout << T.matrix() << std::endl;
		return true;
	}

	static bool CoorTransform(std::vector<Eigen::Vector3d>& points, const Eigen::Vector4d& plane, Eigen::Isometry3d& T)
	{
		// 平面
		double a, b, c, d;
		a = plane[0];
		b = plane[1];
		c = plane[2];
		d = plane[3];
		Eigen::Vector3d o_point, origin, newXaxis, temp1;
		double x = 0, y = 0, z = 0;
		for (auto it : points)
		{
			x += it[0];
			y += it[1];
			z += it[2];
		}
		o_point = Eigen::Vector3d(x / points.size(), y / points.size(), z / points.size());
		Eigen::Vector3d newZaxis(a, b, c);// 新的Z轴
		if (!CalLinePlaneIntersecPoint(a, b, c, d, o_point, newZaxis, origin))
			return false;
		if (!CalLinePlaneIntersecPoint(a, b, c, d, points[0], newZaxis, temp1))
			return false;
		newXaxis = temp1 - origin;// 新的X轴

		Eigen::Vector3d newYaxis = newZaxis.cross(newXaxis);// 新的Y轴

		// 归一化
		newXaxis.normalize();
		newYaxis.normalize();
		newZaxis.normalize();

		// 构造变换矩阵
		Eigen::Matrix3d R;
		R.col(0) = newXaxis;
		R.col(1) = newYaxis;
		R.col(2) = newZaxis;

		T = Eigen::Isometry3d::Identity();
		T.rotate(R);
		T.pretranslate(origin);
		//std::cout << T.matrix() << std::endl;

		T = T.inverse();
		//std::cout << T.matrix() << std::endl;

		// 变换点坐标
		for (auto& it : points)
			it = T * it;

		return true;
	}

	static bool CoorTransform(std::vector<Eigen::Vector3d>& points, Eigen::Isometry3d& T)
	{
		std::vector<cv::Point3d> cvPoints;
		for (auto point : points)
		{
			cv::Point3d tempPt;
			Utils::eigen2cv(point, tempPt);
			cvPoints.push_back(tempPt);
		}
		// 拟合平面
		FitPlane plane(cvPoints, FITMODE::CeresSolve);
		std::vector<double> planeCoeff = plane.getPlaneCoeff();

		double a, b, c, d;
		/*if (!FitPlaneByPoints(points, a, b, c, d))
			return false;*/
		a = planeCoeff[0];
		b = planeCoeff[1];
		c = planeCoeff[2];
		d = planeCoeff[3];
		Eigen::Vector3d newZaxis(a, b, c);// 新的Z轴
		Eigen::Vector3d origin, newXaxis, temp1;
		if (!CalLinePlaneIntersecPoint(a, b, c, d, points[0], newZaxis, origin))
			return false;
		if (!CalLinePlaneIntersecPoint(a, b, c, d, points[1], newZaxis, temp1))
			return false;
		newXaxis = temp1 - origin;// 新的X轴

		Eigen::Vector3d newYaxis = newZaxis.cross(newXaxis);// 新的Y轴

		// 归一化
		newXaxis.normalize();
		newYaxis.normalize();
		newZaxis.normalize();

		// 构造变换矩阵
		Eigen::Matrix3d R;
		R.col(0) = newXaxis;
		R.col(1) = newYaxis;
		R.col(2) = newZaxis;

		T = Eigen::Isometry3d::Identity();
		T.rotate(R);
		T.pretranslate(origin);
		//std::cout << T.matrix() << std::endl;

		T = T.inverse();
		//std::cout << T.matrix() << std::endl;

		// 变换点坐标
		for (auto& it : points)
			it = T * it;

		return true;
	}
}