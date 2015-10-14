#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>


using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	VideoCapture video(CV_CAP_ANY);
	if(!video.isOpened()){
		cout<<"Video not found!"<<endl;
		return 0;
	}
	
	Mat frame;
	char key=0;
	int frame_id=0;

	cv_bridge::CvImage message;
    

	ros::init(argc, argv, "talker");
	ros::NodeHandle n;
	image_transport::ImageTransport it(n);
	image_transport::Publisher publisher;
	publisher=it.advertise("chatter", 1);
	ros::Rate loop_rate(10);
	
	while (ros::ok() && key!=27)
	{
	    
		video>>frame;
		
		message.header.seq = frame_id;
	    message.header.frame_id = frame_id;
	    message.header.stamp = ros::Time::now();
	    message.encoding = sensor_msgs::image_encodings::BGR8;
	    message.image = frame;
	    publisher.publish(message.toImageMsg());
	    frame_id++;
	    imshow("VideoCam", frame);
		key=cvWaitKey(33);
	    ros::spinOnce();
	    loop_rate.sleep();	    
  	}

  	video.release();

	return 0;
}