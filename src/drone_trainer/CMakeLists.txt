cmake_minimum_required(VERSION 3.8)
project(drone_trainer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ros_gz_interfaces REQUIRED)

include_directories(include)

add_executable(run_all 
  src/main.cpp 
  src/trainer.cpp
)

ament_target_dependencies(run_all 
  rclcpp 
  geometry_msgs 
  nav_msgs 
  std_msgs 
  tf2 
  tf2_geometry_msgs
  ros_gz_interfaces
)

install(TARGETS
  run_all
  DESTINATION lib/${PROJECT_NAME})

# Install headers (optional but good practice)
install(DIRECTORY include/
  DESTINATION include
)

ament_package()