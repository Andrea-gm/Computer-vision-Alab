from pathlib import Path

from setuptools import find_packages, setup

package_name = "agent_10_ensemble"
here = Path(__file__).parent

model_files = []
verifier = here / "models" / "verifier.pkl"
if verifier.is_file():
    model_files = ["models/verifier.pkl"]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/vial_detector.launch.py"]),
        ("share/" + package_name + "/config", ["config/ensemble_params.yaml"]),
        ("share/" + package_name + "/models", model_files),
    ],
    install_requires=["setuptools", "numpy", "opencv-python", "scikit-learn"],
    zip_safe=True,
    maintainer="Jessica Camera",
    maintainer_email="devnull@example.com",
    description="Multi-stage ensemble vial detector (Agent #10)",
    license="MIT",
    extras_require={"test": ["pytest"], "yolo": ["ultralytics"]},
    entry_points={
        "console_scripts": [
            "vial_detector_node = agent_10_ensemble.vial_detector_node:main",
        ],
    },
)
