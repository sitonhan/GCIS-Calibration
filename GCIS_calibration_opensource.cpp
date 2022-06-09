//Usage: 
//	please link Python to this project
//	compile and run in release mode

#include <iostream>
#include "GCIS.h"

int main()
{
	std::string path;

	std::cout << "input path: \n";
	std::cin >> path;

	std::cout << "Load :"<< path<<" \n";
	GCIS gcis(path);
	std::cout << "Calibration... \n";
	gcis.completeCalibration();
	std::cout << "Calibration completed! \n \n" <<
		"starting error evaluate... \n";
	gcis.errorEvaluate(path);
	system("pause");
	return 1;
}