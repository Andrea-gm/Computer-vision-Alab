from setuptools import find_packages, setup

package_name = "agent_06_segmentation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "scripts"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/vial_segment.launch.py"]),
        ("share/" + package_name + "/config", ["config/params.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "opencv-python", "ultralytics", "PyYAML"],
    zip_safe=True,
    maintainer="Jessica Camera",
    maintainer_email="devnull@example.com",
    description="YOLOv8-seg vial instance segmentation ROS2 node (competition agent 06)",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "vial_detector_node = agent_06_segmentation.vial_detector_node:main",
        ],
    },
)
