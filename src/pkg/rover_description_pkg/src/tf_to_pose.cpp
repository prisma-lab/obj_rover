#include <memory>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"  // Per convertire le trasformazioni

using namespace std::chrono_literals;

class TransformPosePublisher : public rclcpp::Node
{
public:
  TransformPosePublisher()
  : Node("transform_pose_publisher")
  {
    // Creiamo un publisher per PoseStamped
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("rover_tf_pose", 1);

    // Creiamo il tf buffer e il listener
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Timer: ogni 100 ms esegue il callback per pubblicare la trasformazione
    timer_ = this->create_wall_timer(
      100ms, std::bind(&TransformPosePublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    geometry_msgs::msg::TransformStamped transformStamped;
    try {
      // Otteniamo la trasformazione da "map" a "base_link"
      transformStamped = tf_buffer_->lookupTransform("rover/map", "rover/base_link", tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "Impossibile ottenere la trasformazione: %s", ex.what());
      return;
    }

    // Convertiamo la trasformazione in una PoseStamped
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = "rover/map"; // Il frame target
    pose_msg.pose.position.x = transformStamped.transform.translation.x;
    pose_msg.pose.position.y = transformStamped.transform.translation.y;
    pose_msg.pose.position.z = transformStamped.transform.translation.z;
    pose_msg.pose.orientation = transformStamped.transform.rotation;

    // Pubblica la pose
    pose_pub_->publish(pose_msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TransformPosePublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
