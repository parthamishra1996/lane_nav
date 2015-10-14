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
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(  ${catkin_INCLUDE_DIRS}  ${OpenCV_INCLUDE_DIRS} )

## Declare a C++ executable
add_executable(talker src/talker.cpp)
add_executable(listener src/listener.cpp)
##add_executable(lane_detect src/listener.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(talker ${catkin_LIBRARIES} )
target_link_libraries(listener ${catkin_LIBRARIES} )
##target_link_libraries(lane_detect ${catkin_LIBRARIES} )
