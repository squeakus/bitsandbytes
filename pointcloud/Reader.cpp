// This code is for reading LAS file and convert it to xyz format

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
//#include "conio.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "Tiffmat.hpp"
//#include "ToolBox.h"

using namespace cv;
using namespace std;

int main ()
{
	ifstream myfile;
	myfile.open("316500_234500.las", ios_base::in|ios_base::binary);
	if(myfile.is_open())
		cout << "The file is open." << "\n" << endl;
	
	// Reading the Offset to data
	unsigned long offset;
	myfile.seekg(96, ios_base::beg);
	myfile.read((char*)&offset, 4);
	printf("Offset to data: %lu \n \n", offset);
	
	// Reading the X scale factor
	double x_scale;
	myfile.seekg(131, ios_base::beg);
	myfile.read((char*)&x_scale, 8);
	printf("X scale factor: %f \n \n", x_scale);

	// Reading the Y scale factor
	double y_scale;
	myfile.seekg(139, ios_base::beg);
	myfile.read((char*)&y_scale, 8);
	printf("Y scale factor: %f \n \n", y_scale);

	// Reading the Z scale factor
	double z_scale;
	myfile.seekg(147, ios_base::beg);
	myfile.read((char*)&z_scale, 8);
	printf("Z scale factor: %f \n \n", z_scale);

	// Reading the X offset
	double x_offset;
	myfile.seekg(155, ios_base::beg);
	myfile.read((char*)&x_offset, 8);
	printf("X offset: %f \n \n", x_offset);
	
	// Reading the Y offset
	double y_offset;
	myfile.seekg(163, ios_base::beg);
	myfile.read((char*)&y_offset, 8);
	printf("Y offset: %f \n \n", y_offset);

	// Reading the Z offset
	double z_offset;
	myfile.seekg(171, ios_base::beg);
	myfile.read((char*)&z_offset, 8);
	printf("Z offset: %f \n \n", z_offset);
	
	
	// Reading the POINT DATA RECORD FORMAT 1
	myfile.seekg(0, ios_base::end);
	unsigned int file_size=myfile.tellg();
	//cout << "File Size: " << file_size << "\n" <<endl;
	unsigned int pdata_size=file_size-offset;
	//cout << "Point Data Record Size: " << pdata_size << "\n" <<endl;
	unsigned int point_records=pdata_size/28;
	cout << "Number of Point records: " << point_records << "\n" <<endl;
	
	myfile.seekg(offset, ios_base::beg);
	Mat points(point_records, 4, CV_64F);
	long x, y, z;
	//double t;
	unsigned short intensity;

	for(int i=0; i<point_records; i++)
	{
		myfile.read((char*)&x, 4);
		myfile.read((char*)&y, 4);
		myfile.read((char*)&z, 4);
		myfile.read((char*)&intensity, 2);
		myfile.seekg(14, ios_base::cur);
		//myfile.seekg(8, ios_base::cur);
		//myfile.read((char*)&t, 8);
		points.at<double>(i, 0)=x*x_scale+x_offset;
		points.at<double>(i, 1)=y*y_scale+y_offset;
		points.at<double>(i, 2)=z*z_scale+z_offset;
		//points.at<double>(i, 3)=t;
		points.at<double>(i, 3)=intensity;
	}

	Size s = points.size();
	int rows = s.height;
	int cols = s.width;
	cout << "rows: " << rows << "cols: " << cols << "\n";
        cout << "M = "<< endl << " "  << points << endl << endl;
//imwrite("mean.tif",points);
	//	Writing the points into TIFF format
	//		Tiffmat::tiffwrite("write.tif", points);


	// Closing the LAS file
	myfile.close ();
	cout << "The file is closed." << endl;	
	return 0;
}
