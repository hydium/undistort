cmake_minimum_required(VERSION 3.10)

project(undistort) #Project name

set(CMAKE_CXX_STANDARD 14) #setting C++ 14 standard
find_package(PkgConfig) #finding pkg-config is a helper tool
find_package(OpenCV REQUIRED )
find_package(Eigen3 REQUIRED )


add_definitions  ( ${EIGEN3_DEFINITIONS} )

#using pkg-config to getting Gstreamer
# pkg_check_modules(GSTREAMER REQUIRED gstreamer-app-1.0 gstreamer-1.0)

#including GStreamer header files directory
include_directories(
        # ${GLIB_INCLUDE_DIRS}
        # ${GSTREAMER_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${EIGEN3_INCLUDE_DIRS}
)

#linking GStreamer library directory
# link_directories(
#         ${GLIB_LIBRARY_DIRS}
#         ${GSTREAMER_LIBRARY_DIRS}
# )

#building target executable
add_executable(${PROJECT_NAME} undistort.cpp)

#linking Gstreamer library with target executable
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
