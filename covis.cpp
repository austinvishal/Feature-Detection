//============================================================================
// Name        : covis.cpp
// Author      : Vishal and Yeshasvi
// Version     :
// Copyright   : Your copyright notice
// Description :  Compares the first frame with the next frame
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <nonfree/features2d.hpp>
#include <calib3d/calib3d.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace localTimeNS = boost::posix_time;
typedef localTimeNS::ptime mTime;
typedef localTimeNS::microsec_clock mClock;
typedef localTimeNS::time_duration mDuration;

using namespace std;
using namespace cv;

int minHessian = 400;   							// initialize globally 

// Compares the first frame with the next frame

int main1() {
	
		Mat frame_bgr, frame_grey_1, frame_grey, descriptors_1, descriptors,  // Declare the Mat variables
			img_keypoints_1, img_keypoints, img_matches, H;
		
		SurfFeatureDetector detector(minHessian);	//Construct the SurffeatureDetector object
		SurfDescriptorExtractor extractor;			//Construct the Surfdescriptor extractor object
		FlannBasedMatcher matcher;					// Construct the Flannbased matcher object
		
		vector<KeyPoint> keypoints_1, keypoints; 	// vector of keypoints
		std::vector < DMatch > matches;  			// vector of matches 
		std::vector < Point2f > obj_corners(4);     //vector of type Point2f  containing object corners
		std::vector < Point2f > scene_corners(4); 	//vector of type Point2f  containing scene corners


	VideoCapture cap("video2.mp4"); 				// open the video file for reading
	if (!cap.isOpened())  							// if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}
	double fps = cap.get(CV_CAP_PROP_FPS); 			//get the frames per seconds of the video
	cout << "Frame per seconds : " << fps << endl;
	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); 	//create a window called "MyVideo"
	namedWindow("My first Image-Greyscale", CV_WINDOW_AUTOSIZE);
	namedWindow("My first Image-Greyscale with keypoints", CV_WINDOW_AUTOSIZE);
	namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);

	
	
	// Read the first frame

	cap.read(frame_bgr); 							// read a new frame from video
	cvtColor(frame_bgr, frame_grey, CV_BGR2GRAY);
	frame_grey_1 = frame_grey.clone();				// stores the first greyscale image
	imshow("My first Image-Greyscale", frame_grey_1);

	//Detect the keypoints for 1st image using SURF Detector
	detector.detect(frame_grey_1, keypoints_1);

	// Draw keypoints detected by SURF detector
	drawKeypoints(frame_grey_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
			DrawMatchesFlags::DEFAULT);

	// Calculates the descriptors
	extractor.compute(frame_grey_1, keypoints_1, descriptors_1);
	imshow("My first Image-Greyscale with keypoints", img_keypoints_1);

	while (1) {

		bool bSuccess = cap.read(frame_bgr); 		// read a new frame from video

		if (!bSuccess) 								//if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}
		cvtColor(frame_bgr, frame_grey, CV_BGR2GRAY);//change it to grayscale
		imshow("MyVideo-Original", frame_bgr); 		//show the frame in "MyVideo" window

		//Detect the keypoints for current image using SURF Detector
		detector.detect(frame_grey, keypoints);

		//-- Draw keypoints
		drawKeypoints(frame_grey, keypoints, img_keypoints, Scalar::all(-1),
				DrawMatchesFlags::DEFAULT);

		// Calculates the descriptors
		extractor.compute(frame_grey, keypoints, descriptors);

		//Matching descriptor vectors using FLANN matcher
		matcher.match(descriptors_1, descriptors, matches);

		double max_dist = 0;
		double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors_1.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;
		}

		std::vector < DMatch > good_matches;

		for (int i = 0; i < descriptors_1.rows; i++) {
			if (matches[i].distance < 3 * max(min_dist, 0.02)) {
				good_matches.push_back(matches[i]);
			}
		}

		//-- Draw matches
		drawMatches(frame_grey_1, keypoints_1, frame_grey, keypoints,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		imshow("MyVideo-Grey with keypoints", img_keypoints); 

		//-- Localize the object

		std::vector < Point2f > obj;
		std::vector < Point2f > scene;
		for (unsigned i = 0; i < good_matches.size(); i++) {
			//-- Get the keypoints from the matches
			obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints[good_matches[i].trainIdx].pt);
		}
		//Calculate transformation between two images
		H = findHomography(obj, scene, CV_RANSAC);
		//Define the bounding box of the bulb
		obj_corners[0] = Point2f(24, 44);
		obj_corners[1] = Point2f(196, 44);
		obj_corners[2] = Point2f(196, 210);
		obj_corners[3] = Point2f(24, 210);
		
		//To Calculate the images of corners of rectangle by transformation
		perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(frame_grey_1.cols, 0),
				scene_corners[1] + Point2f(frame_grey_1.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1] + Point2f(frame_grey_1.cols, 0),
				scene_corners[2] + Point2f(frame_grey_1.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2] + Point2f(frame_grey_1.cols, 0),
				scene_corners[3] + Point2f(frame_grey_1.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3] + Point2f(frame_grey_1.cols, 0),
				scene_corners[0] + Point2f(frame_grey_1.cols, 0),
				Scalar(0, 255, 0), 4);
		
		//--Good matches and object detection
		imshow("Good Matches", img_matches);

		if (waitKey(5) > 0) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
				{
			cout << "esc key is pressed by user" << endl;
			break;
		}

	}

	// Destroy the image window
	cvDestroyWindow("MyVideo-Original");
	cvDestroyWindow("MyVideo-Grey");
	return 0;

}

//============================================================================
// Name        : covis.cpp
// Author      : Vishal and Yeshasvi
// Version     :
// Copyright   : Your copyright notice
// Description : Comparing current frame to the next one without geographical filtering
//============================================================================

// Comparing current frame to the next one without geographical filtering
int main2() {

	VideoCapture cap("video2.mp4");							 // open the video file for reading

	if (!cap.isOpened()) 									 // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	double fps = cap.get(CV_CAP_PROP_FPS); 					//get the frames per seconds of the video
	cout << "Frame per seconds : " << fps << endl;

	//namedWindow("MyVideo-Grey", CV_WINDOW_AUTOSIZE);
	namedWindow("My first Image-Greyscale with keypoints", CV_WINDOW_AUTOSIZE);
	namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);

	Mat frame_bgr, frame_grey, frame_grey_c, frame_grey_p, descriptors_p,  // Declare the Mat variables
				descriptors_c, img_keypoints, img_matches, H;
		
	SurfFeatureDetector detector(minHessian, 4, 2, 0, true); //Construct the SurffeatureDetector object
	SurfDescriptorExtractor extractor;						//Construct the Surfdescriptor extractor object
	FlannBasedMatcher matcher;								// Construct the Flannbased matcher object
		
	vector<KeyPoint> keypoints_p, keypoints_c;				// vector of keypoints
	std::vector < DMatch > matches;							// vector of matches
	std::vector < Point2f > obj_corners(4); 				//vector of type Point2f  containing object corners
	std::vector < Point2f > scene_corners(4);				//vector of type Point2f  containing scene corners
	
	int p = 1;												//Loop counter
	obj_corners[0] = Point2f(24, 44);						//Define the object corners
	obj_corners[1] = Point2f(196, 44);
	obj_corners[2] = Point2f(196, 210);
	obj_corners[3] = Point2f(24, 210);

	// Read the 1st frame
	cap.read(frame_bgr);									// read a new frame from video
	cvtColor(frame_bgr, frame_grey, CV_BGR2GRAY);			// Change into grayscale
	frame_grey_p = frame_grey.clone();						//clone it for processing

	//Detect the keypoints for 1st image using SURF Detector
	detector.detect(frame_grey_p, keypoints_p);

	// Draw keypoints detected by SURF detector
	drawKeypoints(frame_grey_p, keypoints_p, img_keypoints, Scalar::all(-1),
			DrawMatchesFlags::DEFAULT);

	// Calculates the descriptors
	extractor.compute(frame_grey_p, keypoints_p, descriptors_p);

	imshow("My first Image-Greyscale with keypoints", img_keypoints);

	while (1)
		{

		// Current frame
		bool bSuccess2 = cap.read(frame_bgr); 				// read a new frame from video
		if (!bSuccess2) 									//if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}
		cvtColor(frame_bgr, frame_grey_c, CV_BGR2GRAY);		//Change it to grayscale

		//Detect the keypoints for current image using SURF Detector
		detector.detect(frame_grey_c, keypoints_c);
		
		//-- Draw keypoints
		drawKeypoints(frame_grey_c, keypoints_c, img_keypoints, Scalar::all(-1),
				DrawMatchesFlags::DEFAULT);
		
		// Calculates the descriptors
		extractor.compute(frame_grey_c, keypoints_c, descriptors_c);

		//Matching descriptor vectors using FLANN matcher
		matcher.match(descriptors_p, descriptors_c, matches);

		double max_dist = 0;
		double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors_p.rows; i++) 
		{
			double dist = matches[i].distance;
			if (dist < min_dist)
				min_dist = dist;
			if (dist > max_dist)
				max_dist = dist;
		}

		std::vector < DMatch > good_matches;

		for (int i = 0; i < descriptors_p.rows; i++) 
		{
			if (matches[i].distance < 3 * max(min_dist, 0.02))
			{
				good_matches.push_back(matches[i]);
			}
		}

		//-- Draw matches
		drawMatches(frame_grey_p, keypoints_p, frame_grey_c, keypoints_c,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		imshow("MyVideo-Grey avec keypoints", img_keypoints);

		//-- Localize the object

		std::vector < Point2f > obj;
		std::vector < Point2f > scene;
		for (unsigned i = 0; i < good_matches.size(); i++) 
		{
			//-- Get the keypoints from the matches
			obj.push_back(keypoints_p[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_c[good_matches[i].trainIdx].pt);
		}
		//Find homography
		H = findHomography(obj, scene, CV_RANSAC);
		//Calculate perspective transform
		perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(frame_grey_p.cols, 0),
				scene_corners[1] + Point2f(frame_grey_p.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1] + Point2f(frame_grey_p.cols, 0),
				scene_corners[2] + Point2f(frame_grey_p.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2] + Point2f(frame_grey_p.cols, 0),
				scene_corners[3] + Point2f(frame_grey_p.cols, 0),
				Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3] + Point2f(frame_grey_p.cols, 0),
				scene_corners[0] + Point2f(frame_grey_p.cols, 0),
				Scalar(0, 255, 0), 4);

		imshow("Good Matches", img_matches);

		if (waitKey(30) > 0) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
				{
			cout << "esc key is pressed by user" << endl;
			break;
				}
		frame_grey_p = frame_grey_c.clone();				//Update the previous frame as current frame
		obj_corners = scene_corners;						//Update object corners
		keypoints_p = keypoints_c;							//Update previous keypoints
		descriptors_p = descriptors_c.clone();				//Update previous descriptors
		p++;												//Increment counter
	}	

	// Destroy the image window
	cvDestroyWindow("MyVideo-Grey");
	return 0;

}

//============================================================================
// Name        : covis.cpp
// Author      : Vishal and Yeshasvi
// Version     :
// Copyright   : Your copyright notice
// Description : Comparing current frame to the next one with geographical filtering
//============================================================================



 // Compares the current frame to the next frame with geographical filtering
  
 int main3() {

 VideoCapture cap("video2.mp4"); // open the video file for reading

 if (!cap.isOpened())  // if not success, exit program
 {
 cout << "Cannot open the video file" << endl;
 return -1;
 }

 double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
 cout << "Frame per seconds : " << fps << endl;

 //namedWindow("MyVideo-Grey", CV_WINDOW_AUTOSIZE);
 namedWindow("My first Image-Greyscale with keypoints", CV_WINDOW_AUTOSIZE);
 namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);

Mat frame_bgr, frame_grey, frame_grey_c, frame_grey_p, descriptors_p, descriptors_c,
 img_keypoints_c,img_keypoints_p, img_matches, H;
 
 SurfFeatureDetector detector(minHessian,4,2,true,true);
 SurfDescriptorExtractor extractor;
 FlannBasedMatcher matcher;
 
 vector<KeyPoint> keypoints_p, keypoints_p_f, keypoints_c;
 std::vector<DMatch> matches;
  std::vector<Point2f> obj_corners(4);
 std::vector<Point2f> scene_corners(4);
 Point2f center;				// declare the center
 int r;							//declare the radius
 int p=1;						// initialize counter variable
 obj_corners[0] = Point2f(24, 44);
 obj_corners[1] = Point2f(196, 44);
 obj_corners[2] = Point2f(196, 210);
 obj_corners[3] = Point2f(24, 210);

 
 // Read the 1st frame
 cap.read(frame_bgr); // read a new frame from video
 cvtColor(frame_bgr, frame_grey, CV_BGR2GRAY);
 frame_grey_p=frame_grey.clone();

 center=(obj_corners[0]+obj_corners[1]+obj_corners[2]+obj_corners[3])*0.25;
 r=round(norm(center-obj_corners[0]));

 //Detect the keypoints for 1st image using SURF Detector
 detector.detect(frame_grey_p, keypoints_p);

 // Geographical filtering of keypoints detected
 for( unsigned i=0 ; i<keypoints_p.size() ; i++ )
 {
 if(norm(center-keypoints_p[i].pt)<=r)
 keypoints_p_f.push_back(keypoints_p[i]);
 }

 // Draw keypoints detected by SURF detector
 drawKeypoints(frame_grey_p, keypoints_p_f, img_keypoints_p,
 Scalar::all(-1), DrawMatchesFlags::DEFAULT);

 imshow("My first Image-Greyscale with keypoints", img_keypoints_p);

 // Calculates the descriptors
 extractor.compute(frame_grey_p, keypoints_p_f, descriptors_p);

 while (1) {

 vector<KeyPoint> keypoints_c_f;
 // Current frame

 center=(obj_corners[0]+obj_corners[1]+obj_corners[2]+obj_corners[3])*0.25;
 r=round(norm(center-obj_corners[0]));

 bool bSuccess2 = cap.read(frame_bgr); // read a new frame from video
 if (!bSuccess2) //if not success, break loop
 {
 cout << "Cannot read the frame from video file" << endl;
 break;
 }
 cvtColor(frame_bgr, frame_grey_c, CV_BGR2GRAY);

 //Detect the keypoints for current image using SURF Detector
 detector.detect(frame_grey_c, keypoints_c);

 // Geographical filtering of keypoints detected
 for( unsigned i=0 ; i<keypoints_c.size() ; i++ )
 {
 if(norm(center-keypoints_c[i].pt)<=r)
 keypoints_c_f.push_back(keypoints_c[i]);
 }

 //-- Draw keypoints
 drawKeypoints(frame_grey_c, keypoints_c_f, img_keypoints_c, Scalar::all(-1),
 DrawMatchesFlags::DEFAULT);

 imshow("MyVideo-Grey with keypoints", img_keypoints_c);

 // Calculates the descriptors
 extractor.compute(frame_grey_c, keypoints_c_f, descriptors_c);

 //Matching descriptor vectors using FLANN matcher
 matcher.match(descriptors_p, descriptors_c, matches);

 double max_dist = 0;
 double min_dist = 100;

 //-- Quick calculation of max and min distances between keypoints
 for (int i = 0; i < descriptors_p.rows; i++) {
 double dist = matches[i].distance;
 if (dist < min_dist)
 min_dist = dist;
 if (dist > max_dist)
 max_dist = dist;
 }

 std::vector<DMatch> good_matches;

 for (int i = 0; i < descriptors_p.rows; i++) {
 if (matches[i].distance < 3 * max(min_dist, 0.02)) {
 good_matches.push_back(matches[i]);
 }
 }
 //-- Draw matches
 drawMatches(frame_grey_p, keypoints_p_f, frame_grey_c, keypoints_c_f,
 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

 //-- Localize the object
 std::vector<Point2f> obj;
 std::vector<Point2f> scene;

 for (unsigned i = 0; i < good_matches.size(); i++) {
 //-- Get the keypoints from the matches
 obj.push_back(keypoints_p_f[good_matches[i].queryIdx].pt);
 scene.push_back(keypoints_c_f[good_matches[i].trainIdx].pt);
 }

 H = findHomography(obj, scene, CV_RANSAC);

 perspectiveTransform(obj_corners, scene_corners, H);

 //-- Draw lines between the corners (the mapped object in the scene - image_2 )
 line(img_matches, scene_corners[0] + Point2f(frame_grey_p.cols, 0),
 scene_corners[1] + Point2f(frame_grey_p.cols, 0),
 Scalar(0, 255, 0), 4);
 line(img_matches, scene_corners[1] + Point2f(frame_grey_p.cols, 0),
 scene_corners[2] + Point2f(frame_grey_p.cols, 0),
 Scalar(0, 255, 0), 4);
 line(img_matches, scene_corners[2] + Point2f(frame_grey_p.cols, 0),
 scene_corners[3] + Point2f(frame_grey_p.cols, 0),
 Scalar(0, 255, 0), 4);
 line(img_matches, scene_corners[3] + Point2f(frame_grey_p.cols, 0),
 scene_corners[0] + Point2f(frame_grey_p.cols, 0),
 Scalar(0, 255, 0), 4);

 imshow("Good Matches", img_matches);

 if (waitKey(30) > 0) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
 {
 cout << "esc key is pressed by user" << endl;
 break;
 }
 frame_grey_p=frame_grey_c.clone();
 descriptors_p=descriptors_c.clone();
 keypoints_p_f=keypoints_c_f;
 obj_corners = scene_corners;
 p++;
 }

 // Destroy the image window
 cvDestroyWindow("MyVideo-Grey");
 return 0;

 }
 

// Program to measure the time of each method 
int main() {
	mTime t1 = mClock::local_time();	// register the time
	main2();							// you can change to main1() to measure time of first method
	mTime t2 = mClock::local_time();
	mDuration d = t2 - t1;
	long timeInMilli = d.total_milliseconds();
	cout << "Time in MilliSeconds " << timeInMilli;
	return 0;
}

