cmake_minimum_required(VERSION 2.8.3)

project(QuadrupedRobot（2026） V1.2-URDF-20260508-V3)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
