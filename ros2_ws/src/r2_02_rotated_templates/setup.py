from pathlib import Path

from setuptools import find_packages, setup

package_name = "r2_02_rotated_templates"
_pkg_root = Path(__file__).parent
_template_dir = _pkg_root / "data" / "templates"
_template_files = (
    sorted(
        str(p.relative_to(_pkg_root)).replace("\\", "/")
        for p in _template_dir.glob("*")
        if p.is_file()
    )
    if _template_dir.is_dir()
    else []
)

_data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    ("share/" + package_name + "/launch", ["launch/vial_detect.launch.py"]),
    ("share/" + package_name + "/config", ["config/params.yaml"]),
]
if _template_files:
    _data_files.append(("share/" + package_name + "/data/templates", _template_files))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "scripts"]),
    data_files=_data_files,
    install_requires=["setuptools", "numpy", "opencv-python", "PyYAML"],
    zip_safe=True,
    maintainer="Jessica Camera",
    maintainer_email="devnull@example.com",
    description="Multi-angle rotated template vial detector (R2 agent 02)",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "vial_detector_node = r2_02_rotated_templates.vial_detector_node:main",
        ],
    },
)
