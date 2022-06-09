#pragma once

#include "ELM.h"
#include "ceres/rotation.h"
#include "ceres/ceres.h"
#include <opencv2/core/core.hpp>
using namespace std;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;

//线性求解初值
/*
O: 直线的控制点
P: 直线的方向向量
S: 空间点
*/
static void gFiore(const cv::Mat& O, const cv::Mat& P, const cv::Mat& S, cv::Mat& rt)
{
	int n = S.rows;
	cv::Mat e = cv::Mat::ones(n, 1, CV_64FC1);
	cv::Mat Se;
	cv::hconcat(S, e, Se);
	cv::Mat W, V, Vt;
	cv::SVDecomp(Se, W, V, Vt, cv::SVD::FULL_UV);
	cv::Mat V2 = V(cv::Rect(cv::Point2i(4, 0), cv::Point2i(V.cols, V.rows))).clone();
	cv::Mat D;
	cv::Mat tempP = (P*P.t());
	cv::Mat tempV2 = (V2*V2.t());
	cv::multiply(tempP, tempV2, D);
	cv::Mat tempb = (V2*V2.t()*O*P.t());
	cv::Mat b = -1 * tempb.diag();
	cv::Mat v = D.inv()*b;
	cv::Mat Z = cv::Mat::diag(v);
	cv::Mat II = cv::Mat::eye(n, n, CV_64FC1) - (e*e.t() / n);
	cv::Mat temp = (Z*P + O).t()*II*S;
	cv::SVDecomp(temp, W, V, Vt, cv::SVD::FULL_UV);
	cv::Mat tempdet = (cv::Mat_<double>(3, 1) << 1, 1, cv::determinant(V*Vt.t()));
	cv::Mat R = V * cv::Mat::diag(tempdet)*(Vt/*.t()*/);
	cv::Mat A = Z * P + O;
	cv::Mat AR = A * R;
	cv::Mat c;
	cv::reduce(S - AR, c, 0, cv::REDUCE_AVG);
	cv::Mat t = -1 * R*(c.t());
	cv::hconcat(R, t, rt);
}

//先全局优化
struct CostFunctorPnP
{
	CostFunctorPnP(cv::Point3d pt, cv::Mat line)
	{
		_m = line.at<double>(0);
		_n = line.at<double>(1);
		_p = line.at<double>(2);
		_x1 = line.at<double>(3);
		_y1 = line.at<double>(4);
		_z1 = line.at<double>(5);
		_x = pt.x;
		_y = pt.y;
		_z = pt.z;
	}

	template <typename T>
	bool operator()(const T* const r, const T* const t, T* residual) const
	{
		T angleaxis[3];
		/*ceres::RotationMatrixToAngleAxis<T>(r, angleaxis);*/
		angleaxis[0] = r[0];
		angleaxis[1] = r[1];
		angleaxis[2] = r[2];
		//变换后空间坐标
		T pt[3];
		T p[3];
		pt[0] = T(_x);
		pt[1] = T(_y);
		pt[2] = T(_z);
		ceres::AngleAxisRotatePoint(angleaxis, pt, p);
		p[0] += t[0];
		p[1] += t[1];
		p[2] += t[2];

		//求（x,y,z）到直线（_m,_n,_p,_x1,_y1,_z1）距离
		T l = (_m * (p[0] - _x1) + _n * (p[1] - _y1) + _p * (p[2] - _z1)) / (_m * _m + _n * _n + _p * _p);
		T xc = _m * l + _x1;
		T yc = _n * l + _y1;
		T zc = _p * l + _z1;
		residual[0] = (p[0] - xc) * (p[0] - xc) + (p[1] - yc) * (p[1] - yc) + (p[2] - zc) * (p[2] - zc);
		return true;
	}

	double _m, _n, _p, _x1, _y1, _z1, _x, _y, _z;
};
static void Ceres4PnP(std::vector<cv::Point3d> pts, std::vector<cv::Mat> lines, cv::Mat& rt)
{
	// 寻优参数的初值
	cv::Mat rvec;
	cv::Rodrigues(rt(cv::Rect(0, 0, 3, 3)), rvec);
	double r[3] = { rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2) };
	double t[3] = { rt.at<double>(0,3),rt.at<double>(1,3),rt.at<double>(2,3) };

	// 第二部分：构建寻优问题
	ceres::Problem problem;
	for (int i = 0; i < pts.size(); i++)
	{
		//使用自动求导，将之前的代价函数结构体传入，第一个1是输出维度，即残差的维度，后面3，3是输入维度，即待寻优参数的维度。
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<CostFunctorPnP, 1, 3, 3>(new CostFunctorPnP(pts[i], lines[i]));
		//向问题中添加误差项
		problem.AddResidualBlock(cost_function, nullptr, r, t);
	}


	//第三部分： 配置并运行求解器
	ceres::Solver::Options options;

	options.minimizer_type = ceres::TRUST_REGION;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::DENSE_QR;
	options.use_nonmonotonic_steps = true;

	//options.minimizer_type = ceres::LINE_SEARCH;
	//options.max_lbfgs_rank = 30;/*20*/
	//options.max_num_line_search_step_size_iterations = 100;

	//配置增量方程的解法
	//options.linear_solver_type = ceres::DENSE_SCHUR;
	//设置迭代次数
	options.max_num_iterations = 200;
	//输出到cout
	options.minimizer_progress_to_stdout = true;
	//优化信息
	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);//求解!!!
	std::cout << summary.BriefReport() << "\n";//输出优化的简要信息
  //最终结果
	cv::Mat Rvecmat(1, 3, CV_64FC1, r);
	cv::Mat R;
	cv::Rodrigues(Rvecmat, R);
	rt = (cv::Mat_<double>(3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t[0],
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t[1],
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t[2]);
}

//再单独优化t
//struct CostFunctorPnPonlyt
//{
//	CostFunctorPnPonlyt(cv::Point3d pt, cv::Mat line, double *r)
//	{
//		_m = line.at<double>(0);
//		_n = line.at<double>(1);
//		_p = line.at<double>(2);
//		_x1 = line.at<double>(3);
//		_y1 = line.at<double>(4);
//		_z1 = line.at<double>(5);
//		_x = pt.x;
//		_y = pt.y;
//		_z = pt.z;
//		_r[0] = r[0]; _r[1] = r[1]; _r[2] = r[2];
//		_r[3] = r[3]; _r[4] = r[4]; _r[5] = r[5];
//		_r[6] = r[6]; _r[7] = r[7]; _r[8] = r[8];
//	}
//
//	template <typename T>
//	bool operator()(/*const T* const r,*/ const T* const t, T* residual) const
//	{
//		/*T* angleaxis;
//		ceres::RotationMatrixToAngleAxis<T>(r, angleaxis);*/
//		//变换后空间坐标
//		T x0 = _r[0] * _x + _r[1] * _y + _r[2] * _z + t[0];
//		T y0 = _r[3] * _x + _r[4] * _y + _r[5] * _z + t[1];
//		T z0 = _r[6] * _x + _r[7] * _y + _r[8] * _z + t[2];
//		//求（x1,y1,z1）到平面（_m,_n,_p,_x0,_y0,_z0）距离
//		T l = (_m * (x0 - _x1) + _n * (y0 - _y1) + _p * (z0 - _z1)) / (_m * _m + _n * _n + _p * _p);
//		T xc = _m * l + _x1;
//		T yc = _n * l + _y1;
//		T zc = _p * l + _z1;
//		residual[0] = (x0 - xc) * (x0 - xc) + (y0 - yc) * (y0 - yc) + (z0 - zc) * (z0 - zc);
//		return true;
//	}
//
//	double _m, _n, _p, _x1, _y1, _z1, _x, _y, _z;
//	double _r[9];
//};
//static void Ceres4PnPonlyt(std::vector<cv::Point3d> pts, std::vector<cv::Mat> lines, cv::Mat& rt)
//{
//	// 寻优参数的初值
//	double r[9] = { rt.at<double>(0,0),rt.at<double>(0,1),rt.at<double>(0,2),
//	rt.at<double>(1,0) ,rt.at<double>(1,1) ,rt.at<double>(1,2) ,
//	rt.at<double>(2,0) ,rt.at<double>(2,1) ,rt.at<double>(2,2) };
//	double t[3] = { rt.at<double>(0,3),rt.at<double>(1,3),rt.at<double>(2,3) };
//
//	// 第二部分：构建寻优问题
//	ceres::Problem problem;
//	for (int i = 0; i < pts.size(); i++)
//	{
//		//使用自动求导，将之前的代价函数结构体传入，第一个1是输出维度，即残差的维度，后面9，3是输入维度，即待寻优参数的维度。
//		ceres::CostFunction* cost_function =
//			new ceres::AutoDiffCostFunction<CostFunctorPnPonlyt, 1, /*9,*/ 3>(new CostFunctorPnPonlyt(pts[i], lines[i], r));
//		//向问题中添加误差项
//		problem.AddResidualBlock(cost_function, nullptr, /*r,*/ t);
//	}
//
//	//第三部分： 配置并运行求解器
//	ceres::Solver::Options options;
//	options.minimizer_type = ceres::TRUST_REGION;
//	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//	options.linear_solver_type = ceres::DENSE_QR;
//	options.use_nonmonotonic_steps = true;
//	options.min_relative_decrease = 1e-5;
//	options.max_num_iterations = 100;
//
//	//输出到cout
//	options.minimizer_progress_to_stdout = true;
//	//优化信息
//	ceres::Solver::Summary summary;
//	Solve(options, &problem, &summary);//求解!!!
//
//	std::cout << summary.BriefReport() << "\n";//输出优化的简要信息
//  //最终结果
//	double result[12] = { r[0], r[1], r[2], t[0], r[3], r[4], r[5], t[1], r[6], r[7], r[8], t[2] };
//	rt = (cv::Mat_<double>(3, 4) << r[0], r[1], r[2], t[0],
//		r[3], r[4], r[5], t[1],
//		r[6], r[7], r[8], t[2]);
//}