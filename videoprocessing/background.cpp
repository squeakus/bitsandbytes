#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
 
int main(int argc, char *argv[])
{
    cv::Mat frame;
    cv::Mat back;
    cv::Mat fore;
    cv::VideoCapture cap("GOPR0072.MP4");
    const int nmixtures =3;
    const bool bShadowDetection = true;
    const int history = 5;
    cv::BackgroundSubtractorMOG2 bg (history,nmixtures,bShadowDetection);
 
    std::vector<std::vector<cv::Point> > contours;
 
    cv::namedWindow("Frame");
    cv::namedWindow("Background");
 
    for(;;)
    {
        cap >> frame;
        bg.operator ()(frame,fore);
        bg.getBackgroundImage(back);
        cv::erode(fore,fore,cv::Mat());
        cv::dilate(fore,fore,cv::Mat());
        cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        cv::drawContours(frame,contours,-1,cv::Scalar(0,0,255),2);
        cv::imshow("Frame",frame);
        cv::imshow("Background",back);
        if(cv::waitKey(30) >= 0) break;
    }
    return 0;
}
