#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cmath>

class ObstacleMover : public rclcpp::Node {
public:
    ObstacleMover() : Node("obstacle_mover") {
        pub_obs1_ = this->create_publisher<geometry_msgs::msg::Twist>("/obstacle1/cmd_vel", 10);
        pub_obs2_ = this->create_publisher<geometry_msgs::msg::Twist>("/obstacle2/cmd_vel", 10);

        pub_enable1_ = this->create_publisher<std_msgs::msg::Bool>("/obstacle1/enable", 10);
        pub_enable2_ = this->create_publisher<std_msgs::msg::Bool>("/obstacle2/enable", 10);
        
        timer_ = rclcpp::create_timer(
            this->get_node_base_interface(),
            this->get_node_timers_interface(),
            this->get_clock(), 
            std::chrono::milliseconds(20), 
            std::bind(&ObstacleMover::move, this)
        );
        start_time_ = this->now();
    }

private:
    void move() {
        std_msgs::msg::Bool enable_msg;
        enable_msg.data = true;
        pub_enable1_->publish(enable_msg);
        pub_enable2_->publish(enable_msg);

        auto now = this->now();
        double t = (now - start_time_).seconds();

        geometry_msgs::msg::Twist cmd1;
        cmd1.linear.z = 0.5 * std::sin(t); // Bob up and down
        cmd1.linear.y = 1.0 * std::cos(t * 0.5); // Sway left/right slow
        pub_obs1_->publish(cmd1);

        geometry_msgs::msg::Twist cmd2;
        cmd2.linear.x = 0.5 * std::cos(t); 
        cmd2.linear.y = 0.8 * std::sin(t);
        pub_obs2_->publish(cmd2);
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_obs1_, pub_obs2_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_enable1_, pub_enable2_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time start_time_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObstacleMover>());
    rclcpp::shutdown();
    return 0;
}