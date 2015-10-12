cmake_minimum_required(VERSION 2.8.3)
project(lane_nav)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  std_msgs
)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cv_bridge image_transport roscpp rospy std_msgs
)

include_directories(  ${catkin_INCLUDE_DIRS}  ${OpenCV_INCLUDE_DIRS} )

## Declare a C++ executable
add_executable(talker src/talker.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(talker ${catkin_LIBRARIES} )
