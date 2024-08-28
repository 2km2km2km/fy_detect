#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <target_dist_calculator/DetectOut.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ImageSplitter
{
public:
    ImageSplitter()
    {
        image_pub_ = nh_.advertise<sensor_msgs::Image>("image_topic", 10);
        detect_out_sub_ = nh_.subscribe("/target_dist_calculator/detect_out", 10, &ImageSplitter::callback, this);
    }

private:
    void callback(const target_dist_calculator::DetectOut::ConstPtr& msg)
    {
        // Convert ROS Image message to OpenCV Image using cv_bridge
        sensor_msgs::Image image_msg = msg->image;
        image_pub_.publish(image_msg);
    }

    ros::NodeHandle nh_;
    ros::Publisher image_pub_;
    ros::Subscriber detect_out_sub_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_splitter");
    ImageSplitter image_splitter;
    ros::spin();
    return 0;
}
