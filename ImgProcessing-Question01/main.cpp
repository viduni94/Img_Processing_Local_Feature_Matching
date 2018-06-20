//
//  main.cpp
//  ImgProcessing-Question01
//
//  Created by Viduni Wickramarachchi on 6/20/18.
//  Copyright © 2018 Viduni Wickramarachchi. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <iostream>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main(int argc, char ** argv) {
    
    //Part 3
    //To display the two images in two windows
    String imageName1("data-dir/Fish/img/0001.jpg");
    String imageName2("data-dir/Fish/img/0199.jpg");
    if(argc > 1)
    {
        imageName1 = argv[1];
        imageName2 = argv[2];
    }
    
    Mat img_1;
    Mat img_2;
    
    img_1 = imread(imageName1, IMREAD_COLOR);
    img_2 = imread(imageName2, IMREAD_COLOR);
    
    if(img_1.empty() || img_2.empty())
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    namedWindow("Display Window 1", WINDOW_AUTOSIZE);
    namedWindow("Display Window 2", WINDOW_AUTOSIZE);
    
    imshow("Display Window 1", img_1);
    imshow("Display Window 2", img_2);
    
    //Part 4
    //To crop the first image
    Mat image1_crop;
    img_1(Rect(134, 55, 60, 88)).copyTo(image1_crop);
    namedWindow("Display Window 3", WINDOW_AUTOSIZE);
    imshow("Display Window 3", image1_crop);
    
    //Part 5
    //SURF Feature Detector
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    
    detector -> detect(img_1, keypoints_1);
    detector -> detect(img_2, keypoints_2);
    
    //Write to csv file
    fstream outputFile1, outputFile2;
    
    outputFile1.open( "features0001.csv", ios::out );
    for( size_t ii = 0; ii < keypoints_1.size( ); ++ii )
        outputFile1 << keypoints_1[ii].pt.x << " " << keypoints_1[ii].pt.y <<std::endl;
    outputFile1.close();
    
    outputFile2.open( "features0199.csv", ios::out );
    for( size_t ii = 0; ii < keypoints_2.size( ); ++ii )
        outputFile2 << keypoints_2[ii].pt.x << " " << keypoints_2[ii].pt.y <<std::endl;
    outputFile2.close();
    
    //Part 6
    //SURF Descriptor Extractor
    Ptr<SURF> extractor = SURF::create();
    Mat descriptors_1, descriptors_2;
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);
    
    //Part 7
    //Matching descriptor vectors with a brute force matcher
    BFMatcher matcher(NORM_L2);
    vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );
    
    //Part 8
    //Draw matches
    Mat output_matching;
    drawMatches( img_1, keypoints_1, image1_crop, keypoints_2, matches, output_matching );
    imshow("Matches", output_matching );

    
    waitKey(0);
    return 0;
}
