cmake_minimum_required(VERSION 3.8)
project(emg_control)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/emg_inference_node.py
  scripts/emg_bridge_node.py
  scripts/emg_monitor.py
  # Importable helper modules (installed alongside the node so it can import them):
  scripts/rcnn_model.py     # CRNN classes — required to joblib.load the CRNN model
  scripts/mains_notch.py    # live causal mains notch used by the inference node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
