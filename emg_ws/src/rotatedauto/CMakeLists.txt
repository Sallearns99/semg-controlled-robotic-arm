cmake_minimum_required(VERSION 3.8)
project(rotatedauto)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/teleop_keyboard.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
