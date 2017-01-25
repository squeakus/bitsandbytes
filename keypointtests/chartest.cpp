// my first program in C++
#include <iostream>
#include <cv.h>
#include <highgui.h>
using namespace cv;

int main()
{
  std::cout << "Hello World!";
  for(int i = 0; i < 500; i++){
    std::cout << i << "\n";
    int j = saturate_cast<uchar>(i);
    std::cout << j << "\n";

  } 
}
