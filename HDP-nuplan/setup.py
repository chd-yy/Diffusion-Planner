import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name="hdp_nuplan",
    version="1.0.0",
    author="Yinan Zheng, Tianyi Tan @ Tsinghua AIR",
    # 自动包含 model/data_process/rl 等子包；否则新增的 RL 模块不会进入非 editable 安装。
    packages=setuptools.find_packages(),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="MIT",
)
