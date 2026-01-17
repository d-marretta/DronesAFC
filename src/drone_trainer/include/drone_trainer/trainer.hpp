#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <ros_gz_interfaces/srv/set_entity_pose.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include "drone_trainer/mlp.hpp"



class DroneTrainer : public rclcpp::Node {
public:
    DroneTrainer();
private:
    enum State { STATE_INIT, STATE_RESET, STATE_WAIT_RESET, STATE_ROLLOUT, STATE_UPDATE };
    State state_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_vel_;
    rclcpp::Client<ros_gz_interfaces::srv::SetEntityPose>::SharedPtr client_reset_;    
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_enable_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::TimerBase::SharedPtr timer_main;

    float pos_x=0, pos_y=0, pos_z=0;
    float current_yaw = 0.0;
    float vel_x=0, vel_y=0, vel_z=0;

    bool has_odom = false;
    float goal_x=10.0, goal_y=4.0, goal_z=3.0; 

    MLP brain_;
    std::vector<std::vector<float>> noise_population_;
    std::vector<float> rewards_;
    int gen_;
    int pop_idx_;
    float current_episode_reward_ = 0.0;
    rclcpp::Time start_time_;
    rclcpp::Time reset_trigger_time_;

    float initial_dist_ = 0.0f;

    // Paper constants
    const float REWARD_C1 = 1000.0f;       // Task reward
    const float REWARD_C2 = 1000.0f;       // Crash penalty
    const float PENALTY_W1 = 500.0f;     // Existential penalty weight
    const float T_MAX_STEPS = 50.0f * EPISODE_TIME; // frequency of control (Hz) * episode time (s)

    bool goal_reached_flag_ = false;
    
    std::mt19937 rng_;
    std::normal_distribution<float> dist_;

    void action_loop();
    void reset_env();
};

#endif