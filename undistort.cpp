#include <opencv2/opencv.hpp>
#include <stdio.h>
// #include <Eigen/Core>
// #include <Eigen/Dense>



using namespace cv;
using namespace std;

int
main (int argc, char *argv[])
{
	cout.precision(17);

	// cout << "OpenCV version : " << CV_VERSION << endl;
	// cout << "Major version : " << CV_MAJOR_VERSION << endl;

	// cout << "Minor version : " << CV_MINOR_VERSION << endl;

	// cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;


	Mat dst;

	Mat src = imread("image_000100.jpg");

	double cameraMatrixArray[3][3] = {{753.349186340502,	0,	1010.31833065182},
									  {0,	753.143587767122,	588.647123579411},
									  {0,	0,	1}};


	Mat cameraMatrix = Mat(3, 3, CV_64F, cameraMatrixArray);

	double distortionCoefficientsArray[5] = {-0.358074811139381, 0.150366096279157,	-0.000239617440106,	-0.001364488806427,	-0.031502910462795};

 	
 	Mat distortionCoefficients = Mat(5, 1, CV_64F, distortionCoefficientsArray);
	// for (int i = 0; i < 5; i++)
	// 	cout << distortionCoefficients.at<double>(1, i) << endl;

	undistort(src, dst, cameraMatrix, distortionCoefficients);



	// cameraMatrix.inv();

	cout << "inverse matrix " << cameraMatrix.inv() << endl; 

	// Matrix<double, 3, 3> A;



	// cv::Rect roi(0, 424, 863, 482);

	// dst = dst(roi);
	
	imwrite("undistorted.png", dst);




    return 0;
}
