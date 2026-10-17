from setuptools import find_packages, setup

package_name = "r2_10_fusion_multiorient"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test", "scripts"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/vial_detect.launch.py"]),
        ("share/" + package_name + "/config", ["config/params.yaml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python",
        "ultralytics",
        "PyYAML",
        "Pillow",
    ],
    zip_safe=True,
    maintainer="Jessica Camera",
    maintainer_email="devnull@example.com",
    description="Multi-orientation vial fusion detector (YOLO-TTA + classical)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "vial_detector_node = r2_10_fusion_multiorient.vial_detector_node:main",
        ],
    },
)
