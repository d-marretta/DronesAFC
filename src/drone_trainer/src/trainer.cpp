#include "drone_trainer/trainer.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <iostream>
#include <thread>
#include <cstdlib>


using namespace std::chrono_literals;

DroneTrainer::DroneTrainer() : Node("drone_trainer"), gen_(0), pop_idx_(0) {
    pub_vel_ = this->create_publisher<geometry_msgs::msg::Twist>("/escaper/cmd_vel", 10);
    pub_enable_ = this->create_publisher<std_msgs::msg::Bool>("/escaper/enable", 10);
    client_reset_ = this->create_client<ros_gz_interfaces::srv::SetEntityPose>("/world/quadcopter/set_pose");
    sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/escaper/odometry", rclcpp::SensorDataQoS(),
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
            pos_x = msg->pose.pose.position.x;
            pos_y = msg->pose.pose.position.y;
            pos_z = msg->pose.pose.position.z;

            tf2::Quaternion q(
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z,
                msg->pose.pose.orientation.w
            );
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            current_yaw = (float)yaw;
            has_odom = true;
        });

    std::random_device rd;
    rng_ = std::mt19937(rd());
    dist_ = std::normal_distribution<float>(0.0, 1.0);

    timer_control_ = this->create_wall_timer(50ms, std::bind(&DroneTrainer::control_loop, this));
    timer_logic_ = this->create_wall_timer(10ms, std::bind(&DroneTrainer::state_machine_loop, this));

    state_ = STATE_INIT;
    RCLCPP_INFO(this->get_logger(), "Trainer Node Initialized.");
}

void DroneTrainer::state_machine_loop() {
    auto now = this->now();

    switch (state_) {
        case STATE_INIT:
            if (!client_reset_->service_is_ready()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                    "Waiting for reset service to be available...");
                return; 
            }
            noise_population_.clear();
            rewards_.clear();
            // Generate noise for the entire population
            for(int i=0; i<POP_SIZE; i++) {
                std::vector<float> noise(brain_.get_weight_count());
                for(float &n : noise) n = dist_(rng_);
                noise_population_.push_back(noise);
            }
            pop_idx_ = 0;
            RCLCPP_INFO(this->get_logger(), "=== GEN %d STARTED ===", gen_);
            state_ = STATE_RESET;
            break;

        case STATE_RESET:
            reset_env();
            reset_trigger_time_ = now;
            state_ = STATE_WAIT_RESET;
            break;

        case STATE_WAIT_RESET:
            if ((now - reset_trigger_time_).seconds() > 2.0) {
                start_time_ = now;
                current_episode_reward_ = 0.0f;
                state_ = STATE_ROLLOUT;
            }
            break;

        case STATE_ROLLOUT:
            // If drone flies too far (>20m), kill episode early
            // if (std::abs(pos_x) > 20.0 || std::abs(pos_y) > 20.0) {
            //     current_episode_reward_ -= 200.0; // Huge penalty
            //     RCLCPP_WARN(this->get_logger(), "Member %d WENT OUT OF BOUNDS! Resetting...", pop_idx_+1);
                
            //     rewards_.push_back(current_episode_reward_);
            //     pop_idx_++;
            //     if (pop_idx_ >= POP_SIZE) state_ = STATE_UPDATE;
            //     else state_ = STATE_RESET;
            //     return;
            // }
            // Just check if time passed, the actual logic is in control_loop
            if ((now - start_time_).seconds() > EPISODE_TIME) {
                rewards_.push_back(current_episode_reward_);
                RCLCPP_INFO(this->get_logger(), 
                    "Member %d/%d | Reward: %.2f | End Pos: [%.2f, %.2f, %.2f] | Goal: [%.2f, %.2f, %.2f]", 
                    pop_idx_+1, POP_SIZE, current_episode_reward_, 
                    pos_x, pos_y, pos_z, 
                    goal_x, goal_y, goal_z);
                pop_idx_++;
                if (pop_idx_ >= POP_SIZE) state_ = STATE_UPDATE;
                else state_ = STATE_RESET;
            }
            break;

        case STATE_UPDATE:
            float mean = 0.0f, std_dev = 0.0f;
            for(float r : rewards_) mean += r;
            mean /= POP_SIZE;
            for(float r : rewards_) std_dev += std::pow(r - mean, 2);
            std_dev = std::sqrt(std_dev / POP_SIZE) + 1e-6f;

            std::vector<float> gradient(brain_.get_weight_count(), 0.0f);
            for(int i=0; i<POP_SIZE; i++) {
                // Standard ES gradient approximation:
                // Sum( Noise_i * (Reward_i - Mean) / StdDev )
                float advantage = (rewards_[i] - mean) / std_dev;
                for(size_t w=0; w<gradient.size(); w++) {
                    gradient[w] += noise_population_[i][w] * advantage;
                }
            }

            // Apply update: W_new = W_old + alpha * (1 / (N*sigma)) * Gradient
            for(size_t w=0; w<brain_.weights.size(); w++) {
                brain_.weights[w] += ALPHA * (gradient[w] / (POP_SIZE * SIGMA));
            }

            RCLCPP_INFO(this->get_logger(), "GEN %d COMPLETE. Avg Reward: %.2f", gen_, mean);
            gen_++;
            state_ = STATE_INIT;
            break;
    }
}

void DroneTrainer::control_loop() {
    if (state_ != STATE_ROLLOUT || !has_odom) return;
        
    std_msgs::msg::Bool enable; enable.data = true;
    pub_enable_->publish(enable);

    float dx_world = goal_x - pos_x;
    float dy_world = goal_y - pos_y;
    float dz = goal_z - pos_z;
    float dist = std::sqrt(dx_world*dx_world + dy_world*dy_world + dz*dz);
    
    if (dist < 0.5f && !goal_reached_flag_) {
        goal_reached_flag_ = true; // Set flag so we only save once per success
        RCLCPP_INFO(this->get_logger(), "GOAL REACHED (%.2fm) !!! Saving model...", dist);
        
        brain_.save("drone_weights.txt"); 
    }

    // Transform to body frame to be rotation invariant
    float cos_yaw = std::cos(-current_yaw);
    float sin_yaw = std::sin(-current_yaw);
    
    float dx_body = dx_world * cos_yaw - dy_world * sin_yaw;
    float dy_body = dx_world * sin_yaw + dy_world * cos_yaw;

    float reward = -dist;
    if (dist < 2.0) reward += 0.5;
    if (dist < 1.0) reward += 2.0;
    
    current_episode_reward_ += reward * 0.05f;
    
    // Inference
    std::vector<float> inputs = {
        std::clamp(dx_body / 5.0f, -1.0f, 1.0f), 
        std::clamp(dy_body / 5.0f, -1.0f, 1.0f), 
        std::clamp(dz / 5.0f, -1.0f, 1.0f),
        0.0f // Placeholder or future Yaw Target
    };

    auto outputs = brain_.forward(inputs, noise_population_[pop_idx_], SIGMA);

    // Output mapping
    geometry_msgs::msg::Twist cmd;
    
    // (-1..1) -> (-1.5 m/s .. 1.5 m/s)
    cmd.linear.x = outputs[0] * 1.5f;

    // (-1..1) -> (-1.5 m/s .. 1.5 m/s)
    cmd.linear.y = outputs[1] * 1.5f;

    // (-1..1) -> (-1.0 m/s .. 1.0 m/s)
    cmd.linear.z = outputs[2] * 1.0f;

    // (-1..1) -> (-2.0 rad/s .. 2.0 rad/s)
    cmd.angular.z = outputs[3] * 2.0f;

    pub_vel_->publish(cmd);
}

// void DroneTrainer::reset_env() {
//     goal_reached_flag_ = false;
//     geometry_msgs::msg::Twist stop;
//     for(int i=0; i<5; i++) {
//             pub_vel_->publish(stop);
//             std::this_thread::sleep_for(10ms);
//     }

//     // Use directly ign service call to reset pose
//     std::string cmd = 
//         "ign service -s /world/quadcopter/set_pose "
//         "--reqtype ignition.msgs.Pose "
//         "--reptype ignition.msgs.Boolean "
//         "--timeout 2000 "
//         "--req 'name: \\\"escaper\\\", position: {x: " + 
//         std::to_string(START_X) + ", y: " + 
//         std::to_string(START_Y) + ", z: " + 
//         std::to_string(START_Z) + "}, orientation: {w: 1.0}' "
//         "> /dev/null 2>&1";
    
//     std::string full_cmd = "/bin/bash -c \"source /opt/ros/humble/setup.bash && " + cmd + "\"";
    
//     // std::cout << "[DEBUG] Executing: " << full_cmd << std::endl;

//     int ret = system(full_cmd.c_str());
//     (void)ret; 

//     // Re-enable motors
//     std_msgs::msg::Bool enable; enable.data = true;
//     pub_enable_->publish(enable);
// }

void DroneTrainer::reset_env() {
    goal_reached_flag_ = false;
    geometry_msgs::msg::Twist stop;
    pub_vel_->publish(stop);
    
    // Prepare the service request
    auto request = std::make_shared<ros_gz_interfaces::srv::SetEntityPose::Request>();
    request->entity.name = "escaper";
    request->pose.position.x = START_X;
    request->pose.position.y = START_Y;
    request->pose.position.z = START_Z;
    request->pose.orientation.w = 1.0;

    // Call service asynchronously 
    if (client_reset_->service_is_ready()) {
        client_reset_->async_send_request(request);
    } else {
        RCLCPP_WARN(this->get_logger(), "Reset service not ready!");
    }

    // Re-enable motors
    std_msgs::msg::Bool enable; enable.data = true;
    pub_enable_->publish(enable);
}

