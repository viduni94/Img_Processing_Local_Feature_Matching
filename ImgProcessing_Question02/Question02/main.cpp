//
//  main.cpp
//  ImgProcessing_Question02
//
//  Created by Viduni Wickramarachchi on 6/21/18.
//  Copyright © 2018 Viduni Wickramarachchi. All rights reserved.
//

#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

/// Global variables

Mat src, src1, src_gray, src_gray1;
Mat dst, dst1, detected_edges, detected_edges1;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

void CannyThreshold(int, void*)
{
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );
    
    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*3, kernel_size );
    
    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);
    
    src.copyTo( dst, detected_edges);
    imshow("Display Window 1", dst);
}

void CannyThreshold1(int, void*)
{
    /// Reduce noise with a kernel 3x3
    blur( src_gray1, detected_edges1, Size(3,3) );
    
    /// Canny detector
    Canny( detected_edges1, detected_edges1, lowThreshold, lowThreshold*3, kernel_size );
    
    /// Using Canny's output as a mask, we display our result
    dst1 = Scalar::all(0);
    
    src1.copyTo( dst1, detected_edges1);
    imshow("Display Window 2", dst1);
}

int main( int argc, char** argv )
{
    
    String imageName1("data-dir/Fish/img/0001.jpg");
    String imageName2("data-dir/Fish/img/0199.jpg");
    if(argc > 1)
    {
        imageName1 = argv[1];
        imageName2 = argv[2];
    }
    
    //Mat img_1;
    //Mat img_2;
    
    /// Load an image
    src = imread(imageName1, IMREAD_COLOR);
    src1 = imread(imageName2, IMREAD_COLOR);
    
    if( !src.data || !src1.data )
    { return -1; }
    
    /// Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );
    dst1.create( src1.size(), src1.type() );
    
    /// Convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );
    cvtColor( src1, src_gray1, CV_BGR2GRAY );
    
    /// Create a window
    namedWindow( "Display Window 1", CV_WINDOW_AUTOSIZE );
    namedWindow( "Display Window 2", CV_WINDOW_AUTOSIZE );
    
    /// Create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold:", "Display Window 1", &lowThreshold, max_lowThreshold, CannyThreshold );
    createTrackbar( "Min Threshold:", "Display Window 2", &lowThreshold, max_lowThreshold, CannyThreshold1 );
    
    /// Show the image
    CannyThreshold(0, 0);
    CannyThreshold1(0, 0);
    
    /// Wait until user exit program by pressing a key
    waitKey(0);
    
    return 0;
}
