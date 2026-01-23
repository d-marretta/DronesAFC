#include "drone_trainer/trainer.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <iostream>
#include <thread>
#include <cstdlib>
#include <chrono>

//using namespace std::chrono_literals;

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

            vel_x = msg->twist.twist.linear.x;
            vel_y = msg->twist.twist.linear.y;
            vel_z = msg->twist.twist.linear.z;

            has_odom = true;
        });
    lidar_readings_.resize(16, MAX_LIDAR_DIST);
    sub_scan_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/escaper/scan", rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        
        for(size_t i=0; i<16 && i < msg->ranges.size(); i++) {
            float r = msg->ranges[i];
            
            if (std::isinf(r) || std::isnan(r)) r = MAX_LIDAR_DIST;
            
            // Clamp to max range
            if (r > MAX_LIDAR_DIST) r = MAX_LIDAR_DIST;
            
            // Store raw value for collision check
            lidar_readings_[i] = r;
        }
    });

    std::random_device rd;
    rng_ = std::mt19937(rd());
    dist_ = std::normal_distribution<float>(0.0, 1.0);
    dist_uniform_ = std::uniform_real_distribution<float>(-1.0, 1.0); 
    dist_uniform_z_ = std::uniform_real_distribution<float>(0.0, 1.0);

    timer_main = rclcpp::create_timer(
        this->get_node_base_interface(),
        this->get_node_timers_interface(),
        this->get_clock(),            
        std::chrono::milliseconds(20),
        std::bind(&DroneTrainer::action_loop, this)
    );
    state_ = STATE_INIT;
    RCLCPP_INFO(this->get_logger(), "Trainer Node Initialized.");
}

void DroneTrainer::action_loop() {
    auto now = this->now();

    switch (state_) {
        case STATE_INIT: {
            if (!client_reset_->service_is_ready()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                    "Waiting for reset service to be available...");
                return; 
            }
            noise_population_.clear();
            rewards_.clear();
            // Generate noise for the entire population
            for(int i=0; i<POP_SIZE / 2; i++) {
                std::vector<float> noise(brain_.get_weight_count());
                for(float &n : noise) n = dist_(rng_);
                noise_population_.push_back(noise);
                
                // Use antithetic sampling
                std::vector<float> anti_noise = noise;
                for(float &n : anti_noise) n = -n;
                noise_population_.push_back(anti_noise);
            }
            pop_idx_ = 0;
            RCLCPP_INFO(this->get_logger(), "=== GEN %d STARTED ===", gen_);
            state_ = STATE_RESET;
            break;
        }
        case STATE_RESET: {
            reset_env();
            reset_trigger_time_ = now;
            state_ = STATE_WAIT_RESET;
            break;
        }
        case STATE_WAIT_RESET: {
            // Keep motors running 
            geometry_msgs::msg::Twist hover_cmd;
            hover_cmd.linear.x = 0.0; 
            hover_cmd.linear.y = 0.0; 
            hover_cmd.linear.z = 0.0;
            hover_cmd.angular.z = 0.0;
            pub_vel_->publish(hover_cmd);
            
            float reset_dx = pos_x - start_x;
            float reset_dy = pos_y - start_y;
            float reset_dz = pos_z - start_z;
            float dist_from_start = std::sqrt(reset_dx*reset_dx + reset_dy*reset_dy + reset_dz*reset_dz);

            // If we are further than 50cm from the spawn point, the reset hasn't finished yet
            if (dist_from_start > 1.0f) {
                if ((now - reset_trigger_time_).seconds() > 2.0) {
                    RCLCPP_WARN(this->get_logger(), 
                        "Stuck at %.2fm (Threshold 0.5m). Forcing retry...", dist_from_start);
                    
                    reset_env();
                    reset_trigger_time_ = now;
                    return; 
                }
                        
                // RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                //    "Waiting for reset... X: %.2f, Y: %.2f, Z: %.2f, Dist: %.2fm", pos_x, pos_y, pos_z, dist_from_start);
                return; 
            }
            // Wait for a short duration to ensure reset is complete
            if ((now - reset_trigger_time_).seconds() > 0.5) {
                if (!has_odom) return;

                start_time_ = now;
                current_episode_reward_ = 0.0f;

                float dx = goal_x - pos_x;
                float dy = goal_y - pos_y;
                float dz = goal_z - pos_z;
                initial_dist_ = std::sqrt(dx*dx + dy*dy + dz*dz);

                state_ = STATE_ROLLOUT;
            }
            break;
        }
        case STATE_ROLLOUT: {
            if (!has_odom) return;
            std_msgs::msg::Bool enable; enable.data = true;
            pub_enable_->publish(enable);

            float dx_world = goal_x - pos_x;
            float dy_world = goal_y - pos_y;
            float dz = goal_z - pos_z;
            float dist = std::sqrt(dx_world*dx_world + dy_world*dy_world + dz*dz);

            std::vector<float> inputs_mlp;

            float min_obs_dist = MAX_LIDAR_DIST;

            for(float range : lidar_readings_) {
                inputs_mlp.push_back(std::clamp(range / MAX_LIDAR_DIST, 0.0f, 1.0f));
                if(range < min_obs_dist) min_obs_dist = range;
            }
            // Existential penalty
            current_episode_reward_ -= (PENALTY_W1 / T_MAX_STEPS);

            bool done = false;
            bool crashed = false;
            // Time expired
            if ((now - start_time_).seconds() > EPISODE_TIME) {
                done = true;
            }
            // Goal reached
            else if (dist < 0.5f) {
                done = true;
                RCLCPP_INFO(this->get_logger(), "GOAL REACHED!");
                brain_.save("drone_weights.txt"); 
            }
            // Crashed
            else if ((pos_z < 0.1f) || (min_obs_dist < 0.3f)) { 
                done = true;
                crashed = true; 
            }

            if (done) {
                float terminal_reward = 0.0f;

                if (crashed) {
                    terminal_reward = -REWARD_C2;
                } else {
                    // C1 * (1 - ||p_T|| / ||p_0||) 
                    float progress_ratio = dist / (initial_dist_ + 1e-5f);
                    terminal_reward = REWARD_C1 * (1.0f - progress_ratio);
                }

                // Add terminal reward to the accumulated existential penalty
                current_episode_reward_ += terminal_reward;

                // Logging and state Transition
                rewards_.push_back(current_episode_reward_);
                RCLCPP_INFO(this->get_logger(), 
                    "Member %d Done. Reward: %.2f (Dist: %.2f -> %.2f) | Start: [%.2f, %.2f, %.2f] | End: [%.2f, %.2f, %.2f] | Goal: [%.2f, %.2f, %.2f]", 
                    pop_idx_+1, 
                    current_episode_reward_, 
                    initial_dist_, 
                    dist,
                    start_x, start_y, start_z, 
                    pos_x, pos_y, pos_z,    
                    goal_x, goal_y, goal_z 
                );
                
                pop_idx_++;
                if (pop_idx_ >= POP_SIZE) state_ = STATE_UPDATE;
                else state_ = STATE_RESET;
                return; // Exit rollout
            }


            // Transform to body frame to be rotation invariant
            float cos_yaw = std::cos(-current_yaw);
            float sin_yaw = std::sin(-current_yaw);
            
            float dx_body = dx_world * cos_yaw - dy_world * sin_yaw;
            float dy_body = dx_world * sin_yaw + dy_world * cos_yaw;
            
            // Inference
            inputs_mlp.push_back(std::clamp(dx_body / MAX_DIST_RANGE, -1.0f, 1.0f));
            inputs_mlp.push_back(std::clamp(dy_body / MAX_DIST_RANGE, -1.0f, 1.0f));
            inputs_mlp.push_back(std::clamp(dz / MAX_DIST_RANGE, -1.0f, 1.0f));
            inputs_mlp.push_back(std::clamp(vel_x / MAX_VEL_RANGE, -1.0f, 1.0f));
            inputs_mlp.push_back(std::clamp(vel_y / MAX_VEL_RANGE, -1.0f, 1.0f));
            inputs_mlp.push_back(std::clamp(vel_z / MAX_VEL_RANGE, -1.0f, 1.0f));
            

            auto outputs = brain_.forward(inputs_mlp, noise_population_[pop_idx_], SIGMA);

            // Output mapping
            geometry_msgs::msg::Twist cmd;
            
            cmd.linear.x = outputs[0] * MAX_LIN_VEL_X;
            cmd.linear.y = outputs[1] * MAX_LIN_VEL_Y;
            cmd.linear.z = outputs[2] * MAX_LIN_VEL_Z;
            cmd.angular.z = outputs[3] * MAX_ANG_VEL_Z;

            pub_vel_->publish(cmd);
            break;
        }
        case STATE_UPDATE: {
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
}


void DroneTrainer::reset_env() {
    
    geometry_msgs::msg::Twist stop;
    stop.linear.x = 0.0; stop.linear.y = 0.0; stop.linear.z = 0.0;
    stop.angular.z = 0.0;
    
    for(int i=0; i<5; i++) {
        pub_vel_->publish(stop);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    goal_x = dist_uniform_(rng_) * ROOM_SIZE_X; 
    goal_y = dist_uniform_(rng_) * ROOM_SIZE_Y;
    goal_z = 1.5f + (dist_uniform_z_(rng_) * (ROOM_SIZE_Z - 1.5f));

    float dist;
    do {
        start_x = dist_uniform_(rng_) * ROOM_SIZE_X;
        start_y = dist_uniform_(rng_) * ROOM_SIZE_Y;
        start_z = 1.5f + (dist_uniform_z_(rng_) * (ROOM_SIZE_Z - 1.5f));

        float dx = start_x - goal_x;
        float dy = start_y - goal_y;
        float dz = start_z - goal_z;
        dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    } while (dist < 3.0f); 

    // Prepare the service request
    auto request = std::make_shared<ros_gz_interfaces::srv::SetEntityPose::Request>();
    request->entity.name = "escaper";
    request->pose.position.x = start_x;
    request->pose.position.y = start_y;
    request->pose.position.z = start_z;
    request->pose.orientation.w = 1.0;
    request->pose.orientation.x = 0.0;
    request->pose.orientation.y = 0.0;
    request->pose.orientation.z = 0.0;

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

