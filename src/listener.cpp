#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <utils.h>

//Program to crop a frame of video
using namespace std;
using namespace cv;

IplImage *frame=NULL;

IplImage *crFrame;
IplImage *half_frame;
IplImage *grey;
IplImage *edgesFrameFrame;
CvMemStorage* houghStorage = cvCreateMemStorage(0);


void crop(IplImage *src, IplImage *dest, CvRect rect)
{
	cvSetImageROI(src,rect);
	cvCopy(src,dest);
	cvResetImageROI(src);
}
struct Lane {
	Lane(){}
	Lane(CvPoint a, CvPoint b, float angle, float kl, float bl): p0(a),p1(b),angle(angle),
		votes(0),visited(false),found(false),k(kl),b(bl) { }

	CvPoint p0, p1;
	int votes;
	bool visited, found;
	float angle, k, b;
};

struct Status {
	Status():reset(true),lost(0){}
	ExpMovingAverage k, b;
	bool reset;
	int lost;
};

Status laneR, laneL;
/*std::vector<Vehicle> vehicles;
std::vector<VehicleSample> samples;*/

#define GREEN CV_RGB(0,255,0)
#define RED CV_RGB(255,0,0)
#define BLUE CV_RGB(255,0,255)
#define PURPLE CV_RGB(255,0,255)
#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30

enum{
	SCAN_STEP = 5,			  // in pixels
	LINE_REJECT_DEGREES = 10, // in degrees
    BW_TRESHOLD = 250,		  // edge response strength to recognize for 'WHITE'
    BORDERX = 10,			  // px, skip this much from left & right borders
	MAX_RESPONSE_DIST = 5,	  // px

	CANNY_MIN_TRESHOLD=1, 
	CANNY_MAX_TRESHOLD=100,

	HOUGH_TRESHOLD = 50,		// line approval vote threshold
	HOUGH_MIN_LINE_LENGTH = 50,	// remove lines shorter than this treshold
	HOUGH_MAX_LINE_GAP = 100,   // join lines to one with smaller than this gaps

	CAR_DETECT_LINES = 4,    // minimum lines for a region to pass validation as a 'CAR'
	CAR_H_LINE_LENGTH = 10  // minimum horizontal line length from car body in px
};

void FindResponses(IplImage *img, int startX, int endX, int y, std::vector<int>& list)
{
    // scans for single response: /^\_

	const int row = y * img->width * img->nChannels;
	unsigned char* ptr = (unsigned char*)img->imageData;

    int step = (endX < startX) ? -1: 1;
    int range = (endX > startX) ? endX-startX+1 : startX-endX+1;

    for(int x = startX; range>0; x += step, range--)
    {
        if(ptr[row + x] <= BW_TRESHOLD) continue; // skip black: loop until white pixels show up

        // first response found
        int idx = x + step;

        // skip same response(white) pixels
        while(range > 0 && ptr[row+idx] > BW_TRESHOLD){
            idx += step;
            range--;
        }

		// reached black again
        if(ptr[row+idx] <= BW_TRESHOLD) {
            list.push_back(x);
        }

        x = idx; // begin from new pos
    }
}

void processSide(std::vector<Lane> lanes, IplImage *edgesFrame, bool right) {

	Status* side = right ? &laneR : &laneL;

	// response search
	int w = edgesFrame->width;
	int h = edgesFrame->height;
	const int BEGINY = 0;
	const int ENDY = h-1;
	const int ENDX = right ? (w-BORDERX) : BORDERX;
	int midx = w/2;
	int midy = h/2;
	unsigned char* ptr = (unsigned char*)edgesFrame->imageData;

	// show responses
	int* votes = new int[lanes.size()];
	for(int i=0; i<lanes.size(); i++) votes[i++] = 0;

	for(int y=ENDY; y>=BEGINY; y-=SCAN_STEP) {
		std::vector<int> rsp;
		FindResponses(edgesFrame, midx, ENDX, y, rsp);

		if (rsp.size() > 0) {
			int response_x = rsp[0]; // use first reponse (closest to screen center)

			float dmin = 9999999;
			float xmin = 9999999;
			int match = -1;
			for (int j=0; j<lanes.size(); j++) {
				// compute response point distance to current line
				float d = dist2line(
						cvPoint2D32f(lanes[j].p0.x, lanes[j].p0.y), 
						cvPoint2D32f(lanes[j].p1.x, lanes[j].p1.y), 
						cvPoint2D32f(response_x, y));

				// point on line at current y line
				int xline = (y - lanes[j].b) / lanes[j].k;
				int dist_mid = abs(midx - xline); // distance to midpoint

				// pick the best closest match to line & to screen center
				if (match == -1 || (d <= dmin && dist_mid < xmin)) {
					dmin = d;
					match = j;
					xmin = dist_mid;
					break;
				}
			}

			// vote for each line
			if (match != -1) {
				votes[match] += 1;
			}
		}
	}

	int bestMatch = -1;
	int mini = 9999999;
	for (int i=0; i<lanes.size(); i++) {
		int xline = (midy - lanes[i].b) / lanes[i].k;
		int dist = abs(midx - xline); // distance to midpoint

		if (bestMatch == -1 || (votes[i] > votes[bestMatch] && dist < mini)) {
			bestMatch = i;
			mini = dist;
		}
	}

	if (bestMatch != -1) {
		Lane* best = &lanes[bestMatch];
		float k_diff = fabs(best->k - side->k.get());
		float b_diff = fabs(best->b - side->b.get());

		bool update_ok = (k_diff <= K_VARY_FACTOR && b_diff <= B_VARY_FACTOR) || side->reset;

		printf("side: %s, k vary: %.4f, b vary: %.4f, lost: %s\n", 
			(right?"RIGHT":"LEFT"), k_diff, b_diff, (update_ok?"no":"yes"));
		
		if (update_ok) {
			// update is in valid bounds
			side->k.add(best->k);
			side->b.add(best->b);
			side->reset = false;
			side->lost = 0;
		} else {
			// can't update, lanes flicker periodically, start counter for partial reset!
			side->lost++;
			if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
				side->reset = true;
			}
		}

	} else {
		printf("no lanes detected - lane tracking lost! counter increased\n");
		side->lost++;
		if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
			// do full reset when lost for more than N frames
			side->reset = true;
			side->k.clear();
			side->b.clear();
		}
	}

	delete[] votes;
}

void processLanes(CvSeq* lines, IplImage* edgesFrame, IplImage* crFrame) {

	// classify lines to left/right side
	std::vector<Lane> left, right;

	for(int i = 0; i < lines->total; i++ )
    {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
		int dx = line[1].x - line[0].x;
		int dy = line[1].y - line[0].y;
		float angle = atan2f(dy, dx) * 180/CV_PI;

		if (fabs(angle) <= LINE_REJECT_DEGREES) { // reject near horizontal lines
			continue;
		}

		// assume that vanishing point is close to the image horizontal center
		// calculate line parameters: y = kx + b;
		dx = (dx == 0) ? 1 : dx; // prevent DIV/0!  
		float k = dy/(float)dx;
		float b = line[0].y - k*line[0].x;

		// assign lane's side based by its midpoint position 
		int midx = (line[0].x + line[1].x) / 2;
		if (midx < crFrame->width/2) {
			left.push_back(Lane(line[0], line[1], angle, k, b));
		} else if (midx > crFrame->width/2) {
			right.push_back(Lane(line[0], line[1], angle, k, b));
		}
    }

	// show Hough lines
	for	(int i=0; i<right.size(); i++) {
		cvLine(crFrame, right[i].p0, right[i].p1, CV_RGB(0, 0, 255), 2);
	}
	
	for	(int i=0; i<left.size(); i++) {
		cvLine(crFrame, left[i].p0, left[i].p1, CV_RGB(255, 0, 0), 2);
	}
	namedWindow("Trial3",CV_WINDOW_NORMAL);
	cvShowImage("Trial3",crFrame);

	processSide(left, edgesFrame, false);
	processSide(right, edgesFrame, true);

	// show computed lanes
	int x = crFrame->width * 0.55f;
	int x2 = crFrame->width;
	cvLine(crFrame, cvPoint(x, laneR.k.get()*x + laneR.b.get()), 
		cvPoint(x2, laneR.k.get() * x2 + laneR.b.get()), CV_RGB(0, 255, 0), 2);

	x = crFrame->width * 0;
	x2 = crFrame->width * 0.45f;
	cvLine(crFrame, cvPoint(x, laneL.k.get()*x + laneL.b.get()), 
		cvPoint(x2, laneL.k.get() * x2 + laneL.b.get()), CV_RGB(0, 255, 0), 2);
	namedWindow("Trial1",CV_WINDOW_NORMAL);
	cvShowImage("Trial1",crFrame);
}
void chatterCallback(const sensor_msgs::ImageConstPtr& msg)
{
	
	
	cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    ROS_INFO("Working");
    //frame->imageData = (char*)cv_ptr->image.data;
    cv::Mat &mat = cv_ptr->image;
    frame = new IplImage(mat);
    CvSize frame_size = cvSize(mat.cols,mat.rows/2);

    crFrame=cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
	half_frame=cvCreateImage(cvSize(frame_size.width/2,frame_size.height), IPL_DEPTH_8U, 3);
	grey=cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
	edgesFrameFrame=cvCreateImage(frame_size, IPL_DEPTH_8U, 1);

    cvPyrDown(frame, half_frame, CV_GAUSSIAN_5x5);
	crop(frame,crFrame,cvRect(0,frame_size.height,frame_size.width,frame_size.height));
	cvCvtColor(crFrame, grey, CV_BGR2GRAY);
	cvSmooth(grey, grey, CV_GAUSSIAN, 5, 5);
	
	cvCanny(grey, edgesFrameFrame, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

	double rho = 1;
	double theta = CV_PI/180;
	CvSeq* lines = cvHoughLines2(edgesFrameFrame, houghStorage, CV_HOUGH_PROBABILISTIC, 
		rho, theta, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

	processLanes(lines, edgesFrameFrame, crFrame);
	
	cvShowImage("Trial2",edgesFrameFrame);
    cvReleaseImage(&frame);
    /*
    Mat img(*cv_ptr.image);
    frame = new IplImage(img);*/
}

int main(int argc, char *argv[])
{
	/*CvCapture *inVideo = cvCreateFileCapture("/home/partha/Downloads/cvTrial/road.avi");
	if (!inVideo)
		cout<<"\nVideo Input Failed";*/
	ros::init(argc, argv, "listener");

	ros::NodeHandle n;
	image_transport::ImageTransport it(n);
	image_transport::Subscriber sub;
	namedWindow("Trial2",CV_WINDOW_NORMAL);	
	
	
	
	/*CvSize video_size;
	video_size.height=(int)cvGetCaptureProperty(inVideo,CV_CAP_PROP_FRAME_HEIGHT);
	video_size.width=(int)cvGetCaptureProperty(inVideo,CV_CAP_PROP_FRAME_WIDTH);*/
	
	//CvSize frame_size = cvSize(video_size.width,video_size.height/2);
	

	

	//int key_pressed=0;
	//namedWindow("Trial1",CV_WINDOW_NORMAL);
	
	//ros::Rate loop_rate(10);
	//while(key_pressed!=27)
	//{
		//frame=cvQueryFrame(inVideo);
	sub = it.subscribe("chatter", 1, chatterCallback);
	
	ros::spin();
	    
		//key_pressed=cvWaitKey(33);
	//}

	cvReleaseMemStorage(&houghStorage);

	cvReleaseImage(&grey);
	cvReleaseImage(&edgesFrameFrame);
	cvReleaseImage(&crFrame);
	cvReleaseImage(&half_frame);

	//cvReleaseCapture(&inVideo);
	return 0;
}