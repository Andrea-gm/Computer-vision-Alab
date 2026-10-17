from setuptools import find_packages, setup

package_name = "r2_07_depth_anypose"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/depth_anypose.launch.py"]),
        (
            "share/" + package_name + "/config",
            ["config/params.yaml", "config/vial_3d_dims.json"],
        ),
    ],
    install_requires=["setuptools", "numpy", "opencv-python"],
    zip_safe=True,
    maintainer="Jessica Camera",
    maintainer_email="devnull@example.com",
    description="RGB-D any-pose vial geometry (PCA/cylinder) — Round-2 Agent 07",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "depth_anypose_node = r2_07_depth_anypose.depth_anypose_node:main",
        ],
    },
)
