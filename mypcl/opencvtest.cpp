#include<opencv2/highgui/highgui.hpp>
using namespace cv;
 
int main()
{
 
    Mat img = imread("/home/jonathan/Pictures/additivedesigns/possibilities.png",CV_LOAD_IMAGE_COLOR);
    imshow("opencvtest",img);
    waitKey(0);
 
    return 0;
}
