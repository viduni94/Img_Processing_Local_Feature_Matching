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

Mat src, src_gray;
Mat dst, detected_edges;

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


/** @function main */
int main( int argc, char** argv )
{
    String imageName1("data-dir/Fish/img/0001.jpg");
    String imageName2("data-dir/Fish/img/0199.jpg");
    if(argc > 1)
    {
        imageName1 = argv[1];
        imageName2 = argv[2];
    }
    
    Mat img_1;
    Mat img_2;
    
    /// Load an image
    img_1 = imread(imageName1, IMREAD_COLOR);
    img_2 = imread(imageName2, IMREAD_COLOR);
    
    if(img_1.empty() || img_2.empty())
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    namedWindow("Display Window 1", WINDOW_AUTOSIZE);
    namedWindow("Display Window 2", WINDOW_AUTOSIZE);
    
    /// Create a matrix of the same type and size as src (for dst)
    dst.create( img_1.size(), img_1.type() );
    
    /// Convert the image to grayscale
    cvtColor( img_1, src_gray, COLOR_BGR2GRAY );
    
    /// Create a window
    namedWindow( "Display Window 1", WINDOW_AUTOSIZE );
    
    /// Create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold:", "Display Window 1", &lowThreshold, max_lowThreshold, CannyThreshold );
    
    /// Show the image
    CannyThreshold(0, 0);
    
    /// Wait until user exit program by pressing a key
    waitKey(0);
    
    return 0;
    }
