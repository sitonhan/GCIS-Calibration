#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "tinyxml2.h"
#include "Utils.h"
#include "TransCoordSys.h"
#include "ELM.h"
#include "PnP.h"

//2d Coded Blocks
typedef std::unordered_map<int/*id*/, std::vector<cv::Point2d>/*8 markers pixel coord*/> CodedMarker;
//3d Coded Blocks
typedef std::unordered_map<int/*id*/, std::vector<cv::Point3d>/*8 points 3d coord*/> CodedPoint;
//Infoemation of Images
typedef std::unordered_map<int/*coded digital value*/, CodedMarker> ImgInfo;
//Infomation of Calibration Board
typedef std::unordered_map<int/*num of positions*/, CodedPoint> PanelInfo;
//Infomation of Homo Mat
typedef std::unordered_map<int/*coded digital value*/, cv::Mat> HomoInfo;


class GCIS
{
public:
	GCIS(std::string path) :_path(path) {
		elm = std::make_shared<ELM>(_path);
	}
	~GCIS(){}


	void matchPtsandImgsGetvPts(std::string ptspath, std::string imgpath, std::string vPtspath);
	void readCameraParameter(std::string CameraParaPath, cv::Mat& K, cv::Mat& distPara);
	void samplePixel(int col, int row, int col_num, int row_num, std::vector<cv::Point2d>& u_v);
	void getvPts(const std::vector<cv::Point2d>& undisted_uv, std::vector<cv::Point3d>& vPts, cv::Mat homomat, cv::Mat Trans);
	void getMatchvPtsandFitLine(std::string vPtspath, std::string Linespath);
	void fitlines(const std::vector<std::vector<cv::Point3d>>& pts, std::vector<cv::Vec6d>& lines);
	void fitline(std::vector<cv::Point3d> pts, cv::Vec6d& line)
	{
		double z_coord = 200;
		int n = pts.size();					//参与拟合的点的个数
		cv::Mat F = cv::Mat::ones(2, n, CV_64F);
		cv::Mat	X = cv::Mat::zeros(n, 1, CV_64F);
		cv::Mat	Y = cv::Mat::zeros(n, 1, CV_64F);
		for (int i = 0; i < n; ++i)
		{
			F.at<double>(0, i) = pts[i].z;
			X.at<double>(i, 0) = pts[i].x;
			Y.at<double>(i, 0) = pts[i].y;
		}
		cv::Mat A = F * F.t();
		cv::Mat b1 = F * X;
		cv::Mat b2 = F * Y;
		cv::Mat X1, X2;
		solve(A, b1, X1, 1);
		solve(A, b2, X2, 1);
		line[0] = X1.at<double>(0, 0);
		line[1] = X2.at<double>(0, 0);
		line[2] = 1;
		line[3] = X1.at<double>(1, 0) + X1.at<double>(0, 0) * z_coord;
		line[4] = X2.at<double>(1, 0) + X2.at<double>(0, 0) * z_coord;
		line[5] = z_coord;
	}
	int encode_d(cv::Vec2i digital)
	{
		return((digital[0] + 50) * 101 + digital[1] + 50);
	}
	cv::Vec2i decode_d(int coded_digital)
	{
		int d1 = coded_digital / 101 - 50;
		int d2 = coded_digital % 101 - 50;
		return cv::Vec2i(d1, d2);
	}
	void errorEvaluate(std::string path);
	void calibrationGCIS(const cv::Mat trainingInput, const cv::Mat trainingOutput);

	void completeCalibration();
	void poseEstimate(const std::vector<cv::Vec4d>& d1d2uv, const std::vector<cv::Point3d>& objpts, cv::Mat& Rt);

private:
	std::string _path;
	std::shared_ptr<ELM> elm;
	cv::Mat _K, _distCoeff;
};

void GCIS::matchPtsandImgsGetvPts(std::string ptspath, std::string imgpath, std::string vPtspath)
{
	std::vector<cv::Point3d> allvpts;
	//读相机参数
	cv::Mat K, distPara;
	readCameraParameter(_path + "/BiCalibData.yml", K, distPara);
	//像素采样
	std::vector<cv::Point2d> uv, undistort_uv;
	samplePixel(1388, 1038, 13, 10, uv);
	//去畸变
	cv::undistortPoints(uv, undistort_uv, K, distPara, cv::noArray(), K);
	cv::Mat mat_undis = cv::Mat(undistort_uv);

	//读ptsxml和imgxml 
	std::vector<ImgInfo> vImgInfo;
	tinyxml2::XMLDocument ImgInfoxml;
	tinyxml2::XMLDocument PtsInfoxml;
	ImgInfoxml.LoadFile((imgpath + "/ImgInfo.xml").c_str());
	PtsInfoxml.LoadFile((ptspath + "/PtsInfo.xml").c_str());
	tinyxml2::XMLElement* imgdoc = ImgInfoxml.RootElement();
	tinyxml2::XMLElement* ptsdoc = PtsInfoxml.RootElement();
	//写vPtsxml
	tinyxml2::XMLDocument vPtsxml;
	tinyxml2::XMLElement* vPtsInfo = vPtsxml.NewElement("HomoInfo");
	vPtsxml.InsertEndChild(vPtsInfo);

	//block
	for (tinyxml2::XMLElement* imgblock = imgdoc->FirstChildElement("block");
		imgblock != nullptr; imgblock = imgblock->NextSiblingElement())
	{
		std::string block(imgblock->GetText());
		for (tinyxml2::XMLElement* ptsblock = ptsdoc->FirstChildElement("block");
			ptsblock != nullptr; ptsblock = ptsblock->NextSiblingElement())
		{
			if (block == std::string(ptsblock->GetText()))
			{
				tinyxml2::XMLElement* vPtsblock = vPtsxml.NewElement("block");
				vPtsblock->SetText(ptsblock->GetText());
				vPtsInfo->InsertEndChild(vPtsblock);

				//position
				for (tinyxml2::XMLElement* imgpos = imgblock->FirstChildElement("position");
					imgpos != nullptr; imgpos = imgpos->NextSiblingElement())
				{

					std::vector<cv::Point3d> vPtsPos;

					std::string pos_str(imgpos->GetText());
					int pos = std::stoi(pos_str);
					for (tinyxml2::XMLElement* ptspos = ptsblock->FirstChildElement("position");
						ptspos != nullptr; ptspos = ptspos->NextSiblingElement())
					{
						/*if (std::stoi(std::string(ptspos->GetText())) == 11)
							break;*/
						if (pos == std::stoi(std::string(ptspos->GetText())))
						{
							tinyxml2::XMLElement* vPtspos = vPtsxml.NewElement("position");
							vPtspos->SetText(pos);
							vPtsblock->InsertEndChild(vPtspos);

							std::string Para_string(ptspos->Attribute("PlanePara"));
							std::vector<double> planePara = Utils::string2doublevec(Para_string);

							// digital
							for (tinyxml2::XMLElement* digital = imgpos->FirstChildElement("digital");
								digital != nullptr; digital = digital->NextSiblingElement())
							{
								//构建对应点的vector
								std::vector<cv::Point2d> img2d;
								std::vector<cv::Point3d> pts3d;

								int d = digital->IntText();
								//遍历id
								for (tinyxml2::XMLElement* imgid = digital->FirstChildElement("id");
									imgid != nullptr; imgid = imgid->NextSiblingElement())
								{
									int id = std::stoi(std::string(imgid->GetText()));
									for (tinyxml2::XMLElement* ptsid = ptspos->FirstChildElement("id");
										ptsid != nullptr; ptsid = ptsid->NextSiblingElement())
									{
										if (id == std::stoi(std::string(ptsid->GetText())))
										{
											//pts
											std::vector<cv::Point3d> temppts3d;
											for (tinyxml2::XMLElement* ptspts = ptsid->FirstChildElement("pts");
												ptspts != nullptr; ptspts = ptspts->NextSiblingElement())
											{
												double x = ptspts->DoubleAttribute("x");
												double y = ptspts->DoubleAttribute("y");
												double z = ptspts->DoubleAttribute("z");
												temppts3d.emplace_back(cv::Point3d(x, y, z));
											}
											//img
											std::vector<cv::Point2d> img2dtemp;
											for (tinyxml2::XMLElement* imgpts = imgid->FirstChildElement("pts");
												imgpts != nullptr; imgpts = imgpts->NextSiblingElement())
											{
												double x = imgpts->DoubleAttribute("x");
												double y = imgpts->DoubleAttribute("y");
												img2dtemp.emplace_back(cv::Point2d(x, y));
											}
											img2d.insert(img2d.end(), img2dtemp.begin(), img2dtemp.end());
											pts3d.insert(pts3d.end(), temppts3d.begin(), temppts3d.end());
										}
									}
								}
								if (img2d.size())
								{

									//去畸变
									std::vector<cv::Point2d> undistImg2d;
									cv::undistortPoints(img2d, undistImg2d, K, distPara, cv::noArray(), K);
									//计算平面拟合坐标变换
									std::vector<cv::Point2d> pts2d;
									cv::Mat Trans3dto2d;
									/*TransSys::FitPlane fp(pts3d,TransSys::FITMODE::CeresSolve);
									std::vector<double> planePara = fp.getPlaneCoeff();*/
									std::vector<Eigen::Vector3d> pts3deigen;
									for (auto pt : pts3d)
									{
										pts3deigen.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
									}
									//Utils::WriteDataToTxt(pts3d, vPtspath + "/pts3d.txt");
									Eigen::Isometry3d TranstoplaneEigen;
									TransSys::CoorTransform(pts3deigen,
										Eigen::Vector4d(planePara[0], planePara[1], planePara[2], planePara[3]),
										TranstoplaneEigen);
									Eigen::Matrix4d tMatEigen = TranstoplaneEigen.matrix();
									cv::Mat TranstoplaneMat = (cv::Mat_<double>(4, 4) <<
										tMatEigen(0, 0), tMatEigen(0, 1), tMatEigen(0, 2), tMatEigen(0, 3),
										tMatEigen(1, 0), tMatEigen(1, 1), tMatEigen(1, 2), tMatEigen(1, 3),
										tMatEigen(2, 0), tMatEigen(2, 1), tMatEigen(2, 2), tMatEigen(2, 3),
										0, 0, 0, 1);
									for (auto pt : pts3deigen)
									{
										pts2d.emplace_back(cv::Point2d(pt[0], pt[1]));
									}
									//计算homo
									cv::Mat Homo = cv::findHomography(undistImg2d, pts2d);
									//Utils::WriteDataToTxt(pts2d, vPtspath + "/pts2d.txt");
									//计算虚拟空间点
									std::vector<cv::Point3d> vPts;
									getvPts(undistort_uv, vPts, Homo, TranstoplaneMat);	//虚拟点坐标系为当前平面坐标系

									//写入XML	
									tinyxml2::XMLElement* vPtsdigital = vPtsxml.NewElement("digital");
									vPtsdigital->SetText(d);
									vPtspos->InsertEndChild(vPtsdigital);
									for (auto& vpts : vPts)
									{
										////y-z坐标对调
										//double y = vpts.y;
										//vpts.y = vpts.z;
										//vpts.z = y;

										tinyxml2::XMLElement* vPtsnode = vPtsxml.NewElement("vPts");
										vPtsnode->SetAttribute("x", vpts.x);
										vPtsnode->SetAttribute("y", vpts.y);
										vPtsnode->SetAttribute("z", vpts.z);
										vPtsdigital->InsertEndChild(vPtsnode);
									}
									allvpts.insert(allvpts.end(), vPts.begin(), vPts.end());
									vPtsPos.insert(vPtsPos.end(), vPts.begin(), vPts.end());
								}
							}
						}
					}
					//Utils::WriteDataToTxt(vPtsPos, vPtspath + "/vPts" + std::to_string(pos) + ".txt");;
				}
			}
		}
	}

	std::string filepath = vPtspath + "/vPtsInfo.xml";
	vPtsxml.SaveFile(filepath.c_str());

	Utils::WriteDataToTxt(allvpts, vPtspath + "/vPts.txt");

}

void GCIS::readCameraParameter(std::string CameraParaPath, cv::Mat& K, cv::Mat& distPara)

{
	cv::FileStorage fs(CameraParaPath, cv::FileStorage::READ, "UTF-8");
	if (fs.isOpened() == false)
		return;
	fs["_intrinsicL"] >> K;
	fs["_distortionL"] >> distPara;
	/*for (int i = 0; i < distMat.rows; ++i)
	{
		distPara.emplace_back(distMat.at<double>(0, i));
	}*/
}

void GCIS::samplePixel(int col, int row, int col_num, int row_num, std::vector<cv::Point2d>& u_v)

{
	//Sample the pixel points of Image plane
	int x_max, y_max, x_step, y_step, x_off, y_off;
	x_max = col;
	y_max = row;
	x_step = x_max / col_num;
	y_step = y_max / row_num;
	x_off = x_max % col_num / 2;
	y_off = y_max % row_num / 2;
	for (int i = 0; i < col_num; i++)
	{
		for (int j = 0; j < row_num; j++)
		{
			u_v.push_back(cv::Point2d(x_off + i * x_step, y_off + j * y_step));
		}
	}
}

void GCIS::getvPts(const std::vector<cv::Point2d>& undisted_uv, std::vector<cv::Point3d>& vPts, cv::Mat homomat, cv::Mat Trans)
{
	cv::Mat src(undisted_uv);
	cv::Mat homosrc;
	cv::convertPointsToHomogeneous(src, homosrc);
	homosrc = homosrc.reshape(1, 0).t();
	cv::Mat vPts2dhomo = homomat * homosrc;
	cv::Mat vPts2d = vPts2dhomo.rowRange(0, 1) / vPts2dhomo.row(2);
	vPts2d.push_back(vPts2dhomo.rowRange(1, 2) / vPts2dhomo.row(2));
	vPts2d.push_back(cv::Mat::zeros(1, vPts2d.cols, CV_64FC1));
	vPts2d.push_back(cv::Mat::ones(1, vPts2d.cols, CV_64FC1));
	cv::Mat vPtsMat = Trans.inv()*vPts2d;
	for (int i = 0; i < vPtsMat.cols; ++i)
	{
		vPts.emplace_back(cv::Point3d(vPtsMat.at<double>(0, i), vPtsMat.at<double>(1, i), vPtsMat.at<double>(2, i)));
	}


	////读相机参数
	//cv::Mat K, distPara;
	//readCameraParameter(cameraPath, K, distPara);
	////像素点采样
	//std::vector<cv::Point2d> v_uv, undistort_uv;
	//SamplePixel(1388, 1038, 13, 10, v_uv);
	////去畸变
	//cv::undistortPoints(v_uv, undistort_uv, K, distPara, cv::noArray(), K);
	//cv::Mat mat_undis = cv::Mat(undistort_uv);
	//mat_undis = mat_undis.reshape(1);
	//mat_undis = mat_undis.t();
	//cv::Mat one = cv::Mat::ones(1, mat_undis.cols, CV_64FC1);
	//mat_undis.push_back(one);
	//QString homoinfo_path = QFileDialog::getOpenFileName(this, tr("Open HomoInfo File"), "F:/");
	//tinyxml2::XMLDocument xml;
	//if (xml.LoadFile(homoinfo_path.toStdString().data()))
	//{
	//	xml.PrintError();
	//	exit(1);
	//}
	//tinyxml2::XMLElement* root = xml.RootElement();
	//std::unordered_map<int, cv::Mat> digitalvsPanelPts;
	//for (tinyxml2::XMLElement* block = root->FirstChildElement("block");
	//	block != NULL;
	//	block = block->NextSiblingElement())
	//{
	//	std::string blockvalue = block->Attribute("block");
	//	for (tinyxml2::XMLElement* position = block->FirstChildElement("position");
	//		position != NULL;
	//		position = position->NextSiblingElement())
	//	{
	//		std::string positionvalue = position->GetText();
	//		tinyxml2::XMLElement* rt = position->FirstChildElement("Rt");
	//		std::string rt_string = rt->GetText();
	//		std::vector<string> rt_v_string = Utils::split(rt_string, " ");
	//		cv::Mat RtMat = (cv::Mat_<double>(4, 4) <<
	//			std::stod(rt_v_string[0]), std::stod(rt_v_string[1]), std::stod(rt_v_string[2]), std::stod(rt_v_string[3]),
	//			std::stod(rt_v_string[4]), std::stod(rt_v_string[5]), std::stod(rt_v_string[6]), std::stod(rt_v_string[7]),
	//			std::stod(rt_v_string[8]), std::stod(rt_v_string[9]), std::stod(rt_v_string[10]), std::stod(rt_v_string[11]),
	//			std::stod(rt_v_string[12]), std::stod(rt_v_string[13]), std::stod(rt_v_string[14]), std::stod(rt_v_string[15]));
	//		/*cv::Mat pos_vpanelmat;*/
	//		for (tinyxml2::XMLElement* homoinfo = position->FirstChildElement("Homoinfo");
	//			homoinfo != NULL;
	//			homoinfo = homoinfo->NextSiblingElement())
	//		{
	//			std::string digital = homoinfo->Attribute("digital");
	//			tinyxml2::XMLElement* homo = homoinfo->FirstChildElement();
	//			std::string homo_str = homo->GetText();
	//			std::vector<string> homo_v_str = Utils::split(homo_str, " ");
	//			cv::Mat homoMat = (cv::Mat_<double>(3, 3) <<
	//				std::stod(homo_v_str[0]), std::stod(homo_v_str[1]), std::stod(homo_v_str[2]),
	//				std::stod(homo_v_str[3]), std::stod(homo_v_str[4]), std::stod(homo_v_str[5]),
	//				std::stod(homo_v_str[6]), std::stod(homo_v_str[7]), std::stod(homo_v_str[8]));
	//			cv::Mat v_panelPts = homoMat * mat_undis;
	//			v_panelPts.push_back(one);
	//			v_panelPts = RtMat.inv() * v_panelPts;
	//			v_panelPts = v_panelPts.rowRange(0, 3).t();
	//			v_panelPts = v_panelPts.reshape(3).t();
	//			if (digitalvsPanelPts.find(std::stod(digital)) == digitalvsPanelPts.end())
	//			{
	//				digitalvsPanelPts.emplace(std::stod(digital), v_panelPts);
	//			}
	//			else
	//			{
	//				digitalvsPanelPts[std::stod(digital)].push_back(v_panelPts);
	//			}
	//		}
	//	}
	//}

	//cv::Mat uv_mat = cv::Mat(v_uv);
	//uv_mat = uv_mat.reshape(1);
	//cv::Mat m_d1d2_uv_abcxyz;
	////拟合直线
	//for (auto per_digi : digitalvsPanelPts)
	//{
	//	cv::Mat d1, d2;
	//	cv::Vec2i d = decode_d(per_digi.first);
	//	d1 = d[0] * cv::Mat::ones(uv_mat.rows, 1, CV_64FC1);
	//	d2 = d[1] * cv::Mat::ones(uv_mat.rows, 1, CV_64FC1);
	//	cv::Mat dataset(per_digi.second.cols, 10, CV_64FC1);
	//	d1.copyTo(dataset.colRange(0, 1));
	//	d2.copyTo(dataset.colRange(1, 2));
	//	uv_mat.copyTo(dataset.colRange(2, 4));
	//	std::vector<cv::Vec6d> lines;
	//	for (size_t col = 0; col < per_digi.second.cols; col++)
	//	{
	//		std::vector<cv::Point3d> pts;
	//		for (int i = 0; i < per_digi.second.rows; i++)
	//		{
	//			pts.push_back(per_digi.second.at<cv::Point3d>(i, col));
	//		}
	//		//Ransac4LineFit lineFit(pts);
	//		//lineFit.SetRansacParam(2, 10, 1, 2, 1);
	//		cv::Vec6d line;
	//		fitline(pts, line);
	//		lines.push_back(std::move(line));
	//	}
	//	cv::Mat m_lines(lines);
	//	m_lines = m_lines.reshape(1, 0);
	//	m_lines.copyTo(dataset.colRange(4, 10));
	//	m_d1d2_uv_abcxyz.push_back(dataset);
	//}
	//////输出
	//Utils::WriteDataToTxt(m_d1d2_uv_abcxyz, "F:/2021.10.23/result2.txt");


}

void GCIS::getMatchvPtsandFitLine(std::string vPtspath, std::string Linespath)
{
	tinyxml2::XMLDocument vPtsxml;
	vPtsxml.LoadFile((vPtspath + "/vPtsInfo.xml").c_str());
	tinyxml2::XMLElement* vPtsInfo = vPtsxml.RootElement();
	//遍历block 构建不同pos下的同名点un_map<int,std::vector<cv::Point3d>>
	std::unordered_map<int/*d*/, std::vector<std::vector<cv::Point3d>>/*vec<pts per pos>*/> linePts;
	for (tinyxml2::XMLElement* blocknode = vPtsInfo->FirstChildElement("block");
		blocknode != nullptr; blocknode = blocknode->NextSiblingElement())
	{
		//遍历position 
		for (tinyxml2::XMLElement* positionnode = blocknode->FirstChildElement("position");
			positionnode != nullptr; positionnode = positionnode->NextSiblingElement())
		{
			/*if (positionnode->IntText() == 1)
			{
				continue;
			}*/
			//遍历d
			for (tinyxml2::XMLElement* digitalnode = positionnode->FirstChildElement("digital");
				digitalnode != nullptr; digitalnode = digitalnode->NextSiblingElement())
			{
				int d = digitalnode->IntText();
				std::vector<cv::Point3d> vptsPerd;
				//遍历vPts
				for (tinyxml2::XMLElement* vPtsnode = digitalnode->FirstChildElement("vPts");
					vPtsnode != nullptr; vPtsnode = vPtsnode->NextSiblingElement())
				{
					double x = vPtsnode->DoubleAttribute("x");
					double y = vPtsnode->DoubleAttribute("y");
					double z = vPtsnode->DoubleAttribute("z");
					vptsPerd.emplace_back(cv::Point3d(x, y, z));
				}
				if (linePts.find(d) == linePts.end())	//没有这个d
				{
					linePts.emplace(d, std::vector<std::vector<cv::Point3d>>());
				}
				linePts[d].emplace_back(vptsPerd);
			}
		}

	}

	std::vector<cv::Point2d> uv;
	samplePixel(1388, 1038, 13, 10, uv);
	cv::Mat uv_mat(uv);
	uv_mat = uv_mat.reshape(1);

	cv::Mat lines_mat;
	cv::Mat d1d2uv_mat;
	for (auto it : linePts)
	{
		std::vector<cv::Vec6d> lines_perd;
		fitlines(it.second, lines_perd);
		cv::Vec2d d1d2 = decode_d(it.first);
		cv::Mat lines_perd_mat(lines_perd);
		lines_mat.push_back(lines_perd_mat.reshape(1));
		cv::Mat d1d2uv_perd_mat(lines_perd_mat.rows, 4, CV_64FC1);
		uv_mat.copyTo(d1d2uv_perd_mat.colRange(2, 4));
		for (int i = 0; i < d1d2uv_perd_mat.rows; ++i)
		{
			d1d2uv_perd_mat.at<double>(i, 0) = d1d2[0];
			d1d2uv_perd_mat.at<double>(i, 1) = d1d2[1];
		}
		d1d2uv_mat.push_back(d1d2uv_perd_mat);
	}

	Utils::WriteDataToTxt(lines_mat, Linespath + "/lines.txt");
	Utils::WriteDataToTxt(d1d2uv_mat, Linespath + "/d1d2uv.txt");
}

void GCIS::fitlines(const std::vector<std::vector<cv::Point3d>>& pts, std::vector<cv::Vec6d>& lines)
{
	std::vector<std::vector<cv::Point3d>> linepts(pts.front().size());
	for (auto per_pts : pts)
	{
		for (int i = 0; i < per_pts.size(); ++i)
		{
			linepts[i].emplace_back(per_pts[i]);
		}
	}
	for (auto per_line : linepts)
	{
		cv::Vec6d line;
		fitline(per_line, line);
		lines.emplace_back(line);
	}
}

void GCIS::errorEvaluate(std::string path)
{
	/*std::cout << "输入num:";*/
	int num = 16;
	/*std::cin >> num;*/
	//读取相机标定参数
	readCameraParameter("./data/BiCalibData.yml", _K, _distCoeff);
	//读标定板坐标
	tinyxml2::XMLDocument PtsInfoxml;
	std::string ptsfile = path + "/PtsInfo.xml";
	if (PtsInfoxml.LoadFile(ptsfile.c_str()))
	{
		PtsInfoxml.PrintError();
		exit(1);
	}
	//读ImgInfo文件
	tinyxml2::XMLDocument ImgInfoxml;
	std::string imgfile = path + "/ImgInfo.xml";
	if (ImgInfoxml.LoadFile(imgfile.c_str()))
	{
		ImgInfoxml.PrintError();
		exit(1);
	}
	//匹配点数据
	std::vector<cv::Vec4d> d1d2uv_vec;
	std::vector<cv::Vec3d> xyz_vec;
	tinyxml2::XMLElement* ImgInfo = ImgInfoxml.RootElement();
	tinyxml2::XMLElement* Imgblock = ImgInfo->FirstChildElement("block");
	for (tinyxml2::XMLElement* Imgpos = Imgblock->FirstChildElement("position");
		Imgpos != nullptr; Imgpos = Imgpos->NextSiblingElement())
	{
		if (Imgpos->IntText() == num || -1 == num)
		{
			tinyxml2::XMLElement* PtsInfo = PtsInfoxml.RootElement();
			tinyxml2::XMLElement* Ptsblock = PtsInfo->FirstChildElement("block");
			for (tinyxml2::XMLElement* Ptspos = Ptsblock->FirstChildElement("position");
				Ptspos != nullptr; Ptspos = Ptspos->NextSiblingElement())
			{
				if (Ptspos->IntText() == Imgpos->IntText())
				{
					for (tinyxml2::XMLElement* digital = Imgpos->FirstChildElement("digital");
						digital != nullptr; digital = digital->NextSiblingElement())
					{
						int d = digital->IntText();
						for (tinyxml2::XMLElement* Imgid = digital->FirstChildElement("id");
							Imgid != nullptr; Imgid = Imgid->NextSiblingElement())
						{
							int id = Imgid->IntText();

							for (tinyxml2::XMLElement* Ptsid = Ptspos->FirstChildElement("id");
								Ptsid != nullptr; Ptsid = Ptsid->NextSiblingElement())
							{
								if (id == Ptsid->IntText())
								{
									for (tinyxml2::XMLElement* Imgpts = Imgid->FirstChildElement("pts");
										Imgpts != nullptr; Imgpts = Imgpts->NextSiblingElement())
									{
										double x = Imgpts->DoubleAttribute("x");
										double y = Imgpts->DoubleAttribute("y");
										d1d2uv_vec.emplace_back(cv::Vec4d(decode_d(d)[0], decode_d(d)[1], x, y));
									}
									for (tinyxml2::XMLElement* Ptspts = Ptsid->FirstChildElement("pts");
										Ptspts != nullptr; Ptspts = Ptspts->NextSiblingElement())
									{
										double x = Ptspts->DoubleAttribute("x");
										double y = Ptspts->DoubleAttribute("y");
										double z = Ptspts->DoubleAttribute("z");
										//xyz_vec.emplace_back(cv::Vec3d(x, z, y));	//对调y-z
										xyz_vec.emplace_back(cv::Vec3d(x, y, z));
									}
								}
							}
						}

					}
					break;
				}
			}
			if (-1 != num)
				break;
		}

	}

	cv::Mat d1d2uv_mat(d1d2uv_vec);
	d1d2uv_mat = d1d2uv_mat.reshape(1).t();
	elm->LoadELM();
	cv::Mat predictlines = elm->PredictLine(d1d2uv_mat, _K, _distCoeff);
	cv::Mat xyz_mat(xyz_vec);
	xyz_mat = xyz_mat.reshape(1).t();
	predictlines = predictlines.t();
	std::cout << "Size of samples to be Evaluate: \n"
		<< "Q:\t" << d1d2uv_mat.size() << "\n"
		<< "Points:\t" << xyz_mat.size() << "\n";
	std::vector<double> distance;
	for (int i = 0; i < xyz_mat.cols; ++i)
	{
		double dis = elm->PtoL(xyz_mat.col(i), predictlines.col(i));
		distance.emplace_back(dis);
	}


	double  max, mean, stddev;
	Utils::calculateStddev(distance, max, mean, stddev);


	std::cout << "error evaluate result:" << std::endl;
	std::cout << "error mean:" << mean << "\t stddev:" << stddev << "\n";

	/*std::vector<double> errorPixel;
	for (int i = 0; i < errorx.size(); ++i)
	{
		double error_temp = cv::norm(cv::Point2f(errorx[i], errory[i]));
		errorPixel.emplace_back(error_temp);
	}
	double mean, max, stddev;
	Utils::calculateStddev(errorPixel, max, mean, stddev);
	std::cout << "mean:" << mean << "\tmax:" << max << "\tstddev:" << stddev << "\n";*/
}

void GCIS::calibrationGCIS(const cv::Mat trainingInput, const cv::Mat trainingOutput)
{
	elm->SetParam(2000, trainingInput, trainingOutput);
	elm->CompleteELM();
}

void GCIS::completeCalibration()
{
	std::cout << _path << std::endl;
	matchPtsandImgsGetvPts(_path, _path, _path);
	getMatchvPtsandFitLine(_path, _path);
	cv::Mat d1d2uv, lines;
	Utils::ReadDataFromTxt(_path + "/d1d2uv.txt", d1d2uv);
	Utils::ReadDataFromTxt(_path + "/lines.txt", lines);
	std::cout << "ELM calculate... \n";
	calibrationGCIS(d1d2uv, lines);
	std::cout << "calibration completed! Write parameter to '"<<_path<<" Parameter/' \n";
}

void GCIS::poseEstimate(const std::vector<cv::Vec4d>& d1d2uv, const std::vector<cv::Point3d>& objpts, cv::Mat& Rt)
{
	cv::Mat d1d2uvmat(d1d2uv);
	d1d2uvmat = d1d2uvmat.reshape(1).t();
	elm->LoadELM();
	readCameraParameter(_path, _K, _distCoeff);
	
	cv::Mat lines = elm->PredictLine(d1d2uvmat, _K, _distCoeff);

	//PnP-Lines_Points
	//计算Rt1初值
	std::vector<cv::Mat>vlines;
	for (auto row = 0; row < lines.rows; ++row)
	{
		vlines.emplace_back(lines.row(row));
	}
	cv::Mat ptsmat(objpts);
	ptsmat = ptsmat.reshape(1);
	cv::Mat RtInit;
	gFiore(lines(cv::Rect(3, 0, 3, lines.rows)),
		lines(cv::Rect(0, 0, 3, lines.rows)), ptsmat, RtInit);
	cv::Mat RtCeres = RtInit.clone();
	//优化Rt1
	Ceres4PnP(objpts, vlines, RtCeres);
	Rt = RtCeres;
}

