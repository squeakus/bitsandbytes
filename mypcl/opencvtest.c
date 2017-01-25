#include<opencv2/highgui/highgui.hpp>
 
int main()
{
    IplImage* img = cvLoadImage("/home/jonathan/Pictures/additivedesigns/possibilities.png",CV_LOAD_IMAGE_COLOR);
    cvNamedWindow("opencvtest",CV_WINDOW_AUTOSIZE);
    cvShowImage("opencvtest",img);
    cvWaitKey(0);
    cvReleaseImage(&img);
    return 0;
}
