from setuptools import find_packages, setup

package_name = "agent_08_depth_geometry"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/depth_geometry.launch.py"]),
        (
            "share/" + package_name + "/config",
            ["config/params.yaml", "config/vial_3d_dims.json"],
        ),
    ],
    install_requires=["setuptools", "numpy", "opencv-python"],
    zip_safe=True,
    maintainer="Jessica Camera",
    maintainer_email="devnull@example.com",
    description="Depth / plane / cylinder geometric vial detector (Agent 08)",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "depth_geometry_node = agent_08_depth_geometry.depth_geometry_node:main",
        ],
    },
)
