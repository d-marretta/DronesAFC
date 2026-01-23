#include "drone_trainer/trainer.hpp"
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <cstdlib>

pid_t pid_gazebo = 0;
pid_t pid_bridge = 0;
pid_t pid_obs = 0;


pid_t start_process_bg(const std::string& command) {
    pid_t pid = fork();
    if (pid == 0) { // Child
        setsid(); 
        std::string full_cmd = "source /opt/ros/humble/setup.bash && " + command;
        execl("/bin/bash", "bash", "-c", full_cmd.c_str(), (char *)0);
        exit(1); 
    }
    return pid;
}

void run_bash_blocking(const std::string& command) {
    std::string full_cmd = "/bin/bash -c 'source /opt/ros/humble/setup.bash && " + command + "'";
    int ret = system(full_cmd.c_str()); 
    (void)ret; 
}

void cleanup(int signum) {
    std::cout << "\n\n[C++] RECEIVED SIGNAL " << signum << ". STARTING CLEANUP..." << std::endl;

    if (pid_obs > 0) {
        std::cout << "[C++] Killing obstacle  (PGID " << pid_obs << ")..." << std::endl;
        kill(-pid_obs, SIGKILL);
        waitpid(pid_obs, nullptr, 0);
    }
    
    if (pid_bridge > 0) {
        std::cout << "[C++] Killing bridge (PGID " << pid_bridge << ")..." << std::endl;
        kill(-pid_bridge, SIGKILL);
        waitpid(pid_bridge, nullptr, 0);
        std::cout << "[C++] Bridge killed." << std::endl;
    }

    if (pid_gazebo > 0) {
        std::cout << "[C++] Killing gazebo (PGID " << pid_gazebo << ")..." << std::endl;
        kill(-pid_gazebo, SIGKILL);
        waitpid(pid_gazebo, nullptr, 0);
        std::cout << "[C++] Gazebo killed." << std::endl;
    }
    
    std::cout << "[C++] Shutdown complete." << std::endl;
    rclcpp::shutdown();
    exit(0); // Exit with 0 to indicate clean shutdown
}



int main(int argc, char **argv) {
    signal(SIGINT, cleanup);
    std::cout << "[C++] STARTING SIMULATION SUITE..." << std::endl;

    
    // -s: server only (headless)
    // -r: run immediately
    // -v 1: minimal logging
    std::string gz_cmd = "ign gazebo -s -r -v 1 world.sdf"; 
    pid_gazebo = start_process_bg(gz_cmd);
    std::cout << "[C++] Gazebo Server Started (PID " << pid_gazebo << ")" << std::endl;

    std::cout << "[C++] Waiting 5s for Gazebo to be ready..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    

    // launch bridge
    std::string bridge_topics = 
        "/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock "
        "/escaper/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist "
        "/escaper/enable@std_msgs/msg/Bool@ignition.msgs.Boolean "
        "/escaper/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry "
        "/escaper/scan@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan "
        "/obstacle1/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist "
        "/obstacle2/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist "
        "/obstacle1/enable@std_msgs/msg/Bool@ignition.msgs.Boolean "
        "/obstacle2/enable@std_msgs/msg/Bool@ignition.msgs.Boolean "
        "/world/quadcopter/set_pose@ros_gz_interfaces/srv/SetEntityPose";
            
    std::string bridge_cmd = "ros2 run ros_gz_bridge parameter_bridge " + bridge_topics;
    pid_bridge = start_process_bg(bridge_cmd);
    std::cout << "[C++] Bridge started (PID " << pid_bridge << ")" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::string obs_cmd = "ros2 run drone_trainer obstacle_mover --ros-args -p use_sim_time:=true"; 
    pid_obs = start_process_bg(obs_cmd);
    std::cout << "[C++] Obstacle controller started (PID " << pid_obs << ")" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // spawn drone
    std::cout << "[C++] Spawning drone..." << std::endl;
    // Uses the ros_gz_sim create tool
    std::string spawn_cmd = "ros2 run ros_gz_sim create -world quadcopter -file escaper.sdf -name escaper -x " + 
                                std::to_string(START_X) + " -y " + 
                                std::to_string(START_Y) + " -z " + 
                                std::to_string(START_Z);
    run_bash_blocking(spawn_cmd);

    std::cout << "[C++] Spawning obstacles..." << std::endl;
    
    std::string spawn_obs1 = "ros2 run ros_gz_sim create -world quadcopter -file obstacle1.sdf -name obstacle1 -x 14.0 -y 10.0 -z 7.0";
    run_bash_blocking(spawn_obs1);

    std::string spawn_obs2 = "ros2 run ros_gz_sim create -world quadcopter -file obstacle2.sdf -name obstacle2 -x 17.0 -y 10.0 -z 7.0";
    run_bash_blocking(spawn_obs2);

    // Run trainer
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DroneTrainer>();
    std::cout << "[C++] Trainer started" << std::endl;
    rclcpp::spin(node);
    
    return 0;
}