from pathlib import Path

from setuptools import find_packages, setup

package_name = "agent_07_hog_svm"
_ROOT = Path(__file__).resolve().parent


def _existing(paths):
    return [str(p.relative_to(_ROOT)).replace("\\", "/") for p in paths if p.is_file()]


_model_files = _existing(sorted((_ROOT / "models").glob("*"))) if (_ROOT / "models").is_dir() else []
_script_files = _existing(
    [
        _ROOT / "scripts" / "train_hog_svm.py",
        _ROOT / "scripts" / "evaluate.py",
    ]
)

_data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    ("share/" + package_name + "/launch", ["launch/vial_detect.launch.py"]),
    ("share/" + package_name + "/config", ["config/params.yaml"]),
]
if _model_files:
    _data_files.append(("share/" + package_name + "/models", _model_files))
if _script_files:
    _data_files.append(("share/" + package_name + "/scripts", _script_files))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=_data_files,
    install_requires=["setuptools", "numpy", "opencv-python", "scikit-learn"],
    zip_safe=True,
    maintainer="Jessica Camera",
    maintainer_email="devnull@example.com",
    description="HOG + Linear SVM vial detector (Agent #07)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "vial_detector_node = agent_07_hog_svm.vial_detector_node:main",
        ],
    },
)
