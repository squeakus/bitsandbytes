// This code is for reading LAS file and convert it to xyz format

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
//#include "conio.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Tiffmat.hpp"
//#include "ToolBox.h"

using namespace cv;
using namespace std;

int main ()
{
	// cout << myfile.tellg() << endl;
	
	ifstream myfile;
	myfile.open("316500_234500.las", ios_base::in|ios_base::binary);
	if(myfile.is_open())
		cout << "The file is open." << "\n" << endl;
	
	//// Reading the File Signature ("LASF")
	//char fs[4];
	//myfile.seekg(0, ios_base::beg);
	//myfile.read(fs, 4);
	//printf("File Signature: %.4s \n \n", fs);
	
	//// Reading the Reserved
	//unsigned long reserved;
	//myfile.seekg(4, ios_base::beg);
	//myfile.read ((char*)&reserved, 4);
	//printf("Reserved: %lu \n \n", reserved);
	
	//// Reading the GUID data 1
	//unsigned long gdata1;
	//myfile.seekg(8, ios::beg);
	//myfile.read ((char*)&gdata1, 4);
	//printf("GIUD data 1: %lu \n \n", gdata1);
	
	//// Reading the GUID data 2
	//unsigned short gdata2;
	//myfile.seekg(12, ios_base::beg);
	//myfile.read ((char*)&gdata2, 2);
	//printf("GIUD data 2: %hu \n \n", gdata2);
	
	//// Reading the GUID data 3
	//unsigned short gdata3;
	//myfile.seekg(14, ios_base::beg);
	//myfile.read ((char*)&gdata3, 2);
	//printf("GIUD data 3: %hu \n \n", gdata3);
	
	//// Reading the GUID data 4
	//unsigned char gdata4[8];
	//myfile.seekg(16, ios_base::beg);
	//myfile.read ((char*)&gdata4, 8);
	//printf("GUID data 4: %hhu %hhu %hhu %hhu %hhu %hhu %hhu %hhu \n \n", gdata4[0], 
	//	gdata4[1], gdata4[2], gdata4[3], gdata4[4], gdata4[5], gdata4[6], gdata4[7]);
		
	//// Reading the Version Major
	//unsigned char vmajor;
	//myfile.seekg(24, ios_base::beg);
	//myfile.read ((char*)&vmajor, 1);
	//printf("Varsion Major: %hhu \n \n", vmajor);
	
	//// Reading the Version Major
	//unsigned char vminor;
	//myfile.seekg(25, ios_base::beg);
	//myfile.read ((char*)&vminor, 1);
	//printf("Varsion Minor: %hhu \n \n", vminor);

	//// Reading the System Identifier
	//char systemid[32];
	//myfile.seekg(26, ios_base::beg);
	//myfile.read(systemid, 32);
	//printf("System Identifier: %.32s \n \n", systemid);
	
	//// Reading the Generating Software
	//char gsoftware[32];
	//myfile.seekg(58, ios_base::beg);
	//myfile.read(gsoftware, 32);
	//printf("Generating Software: %.32s \n \n", gsoftware);
	
	//// Reading  the Flight Date Julian
	//unsigned short flight_date;
	//myfile.seekg(90, ios_base::beg);
	//myfile.read((char*)&flight_date, 2);
	//printf("Flight Date Julian: %hu \n \n", flight_date);
		
	//// Reading the Year
	//unsigned short year;
	//myfile.seekg(92, ios_base::beg);
	//myfile.read((char*)&year, 2);
	//printf("Year: %hu \n \n", year);
	
	//// Reading the Header Size
	//unsigned short header_size;
	//myfile.seekg(94, ios_base::beg);
	//myfile.read((char*)&header_size, 2);
	//printf("Header Size: %hu \n \n", header_size);
	
	// Reading the Offset to data
	unsigned long offset;
	myfile.seekg(96, ios_base::beg);
	myfile.read((char*)&offset, 4);
	printf("Offset to data: %lu \n \n", offset);
	
	//// Reading the Number of variable length records
	//unsigned long variable_lr;
	//myfile.seekg(100, ios_base::beg);
	//myfile.read((char*)&variable_lr, 4);
	//printf("Number of variable length records: %lu \n \n", variable_lr);
	
	//// Reading the Point Data Format ID (0-99 for spec)
	//unsigned char point_data_id;
	//myfile.seekg(104, ios_base::beg);
	//myfile.read((char*)&point_data_id, 1);
	//printf("Point Data Format ID: %hhu \n \n", point_data_id);
	
	//// Reading the Point Data Record Length
	//unsigned short pdrl;
	//myfile.seekg(105, ios_base::beg);
	//myfile.read((char*)&pdrl, 2);
	//printf("Point Data Record Length: %hu \n \n", pdrl);
	
	//// Reading the Number of point records
	//unsigned long nr_points;
	//myfile.seekg(107, ios_base::beg);
	//myfile.read((char*)&nr_points, 4);
	//printf("Number of point records: %lu \n \n", nr_points);
		
	//// Reading the Number of point by return
	//unsigned long points_preturn[5];
	//myfile.seekg(111, ios_base::beg);
	//myfile.read((char*)&points_preturn, 20);
	//printf("Number of points per return 1: %lu \n", points_preturn[0]);
	//printf("Number of points per return 2: %lu \n", points_preturn[1]);
	//printf("Number of points per return 3: %lu \n", points_preturn[2]);
	//printf("Number of points per return 4: %lu \n", points_preturn[3]);
	//printf("Number of points per return 5: %lu \n \n", points_preturn[4]);

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
	
	//// Reading the Max X
	//double max_x;
	//myfile.seekg(179, ios_base::beg);
	//myfile.read((char*)&max_x, 8);
	//printf("Max X: %f \n \n", max_x);
	
	//// Reading the Min X
	//double min_x;
	//myfile.seekg(187, ios_base::beg);
	//myfile.read((char*)&min_x, 8);
	//printf("Min X: %f \n \n", min_x);
	
	//// Reading the Max Y
	//double max_y;
	//myfile.seekg(195, ios_base::beg);
	//myfile.read((char*)&max_y, 8);
	//printf("Max Y: %f \n \n", max_y);
	
	//// Reading the Min Y
	//double min_y;
	//myfile.seekg(203, ios_base::beg);
	//myfile.read((char*)&min_y, 8);
	//printf("Min Y: %f \n \n", min_y);
	
	//// Reading Max Z
	//double max_z;
	//myfile.seekg(211, ios_base::beg);
	//myfile.read((char*)&max_z, 8);
	//printf("Max Z: %f \n \n", max_z);
	
	//// Reading Min Z
	//double min_z;
	//myfile.seekg(219, ios_base::beg);
	//myfile.read((char*)&min_z, 8);
	//printf("Min Z: %f \n \n", min_z);
	//cout << myfile.tellg() << endl << "\n";
	
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

	// Writing the points into TIFF format
	Tiffmat::tiffwrite("write.tif", points);
	
	// Reading the points in TIFF format
	Mat points1=Tiffmat::tiffread("write.tif");
	//ToolBox::printMat( points1 ( Range ( 0, 5 ), Range ( 0, 4 ) ) );


	// Closing the LAS file
	myfile.close ();
	cout << "The file is closed." << endl;
		
	//	_getch ();
	return 0;
}
