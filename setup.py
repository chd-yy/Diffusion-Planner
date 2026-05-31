# 导入 Python 内置的 os 模块。
#
# os 模块提供了与操作系统交互的功能。
# 当前脚本主要使用它完成两项工作：
# 1. 获取当前 setup.py 文件所在的目录；
# 2. 将程序的工作目录切换到 setup.py 所在的目录。
import os


# 导入 setuptools。
#
# setuptools 是 Python 中常用的项目打包和安装工具。
#
# 当用户在终端中执行以下命令时：
#
# pip install .
#
# 或者：
#
# python setup.py install
#
# setuptools 会读取当前文件中的配置信息，
# 然后将 diffusion_planner 项目安装到 Python 环境中。
import setuptools


# Change directory to allow installation from anywhere

# __file__ 表示当前正在执行的 Python 文件路径。
#
# 假设当前 setup.py 文件位于：
#
# /home/user/Diffusion-Planner/setup.py
#
# 那么：
#
# __file__
#
# 通常会对应：
#
# /home/user/Diffusion-Planner/setup.py
#
# os.path.realpath(__file__)：
#   将文件路径转换为绝对路径，并解析可能存在的符号链接。
#
# os.path.dirname(...)：
#   获取文件所在的文件夹路径。
#
# 因此，script_folder 最终保存 setup.py 所在的目录。
#
# 例如：
#
# script_folder = "/home/user/Diffusion-Planner"
script_folder = os.path.dirname(os.path.realpath(__file__))

# 将当前程序的工作目录切换到 setup.py 所在的文件夹。
#
# 这样做的作用是：
# 即使用户不是在项目根目录下执行安装命令，
# 后续相对路径仍然会以 setup.py 所在目录为基准。
#
# 例如，用户在其他目录中执行：
#
# pip install /home/user/Diffusion-Planner
#
# 当前脚本仍然能够正确找到项目文件。
os.chdir(script_folder)


# Installs

# 调用 setuptools.setup() 描述需要安装的 Python 项目。
#
# setup() 中传入的参数称为项目元数据。
#
# 这些信息会告诉 setuptools：
# 1. 项目名称是什么；
# 2. 当前版本是什么；
# 3. 作者是谁；
# 4. 需要安装哪些 Python 包；
# 5. 包所在的目录在哪里；
# 6. 项目适用于哪些 Python 版本和操作系统；
# 7. 项目采用什么许可证。
setuptools.setup(

    # 设置项目的安装名称。
    #
    # 安装完成后，可以在 Python 代码中使用：
    #
    # import diffusion_planner
    #
    # 在 pip 的包管理信息中，
    # 该项目也会显示为 diffusion_planner。
    name="diffusion_planner",

    # 设置当前项目的版本号。
    #
    # 这里使用的是：
    #
    # 主版本号.次版本号.修订号
    #
    # 即：
    #
    # 1.0.0
    version="1.0.0",

    # 设置项目作者信息。
    #
    # 该字段主要用于记录项目的开发者或维护者。
    author="Zheng Yinan, Ruiming Liang, Kexin Zheng @ Tsinghua AIR",

    # 指定需要安装的 Python 包。
    #
    # 这里仅显式列出了：
    #
    # diffusion_planner
    #
    # 这表示 setuptools 会将 diffusion_planner 目录
    # 作为一个 Python 包进行安装。
    #
    # 注意：
    # 如果 diffusion_planner 目录中还存在多级子包，
    # 是否能够被完整安装，需要结合项目目录结构进一步确认。
    #
    # 当前代码严格保留原始写法，没有改为其他自动扫描方式。
    packages=["diffusion_planner"],

    # 设置 Python 包所在目录与文件系统目录之间的对应关系。
    #
    # package_dir 的键为空字符串：
    #
    # ""
    #
    # 表示它描述的是所有顶层 Python 包的基础目录。
    #
    # 值为：
    #
    # "."
    #
    # 表示顶层包位于当前目录。
    #
    # 由于前面已经执行：
    #
    # os.chdir(script_folder)
    #
    # 所以这里的当前目录就是 setup.py 所在目录。
    package_dir={"": "."},

    # 设置项目分类信息。
    #
    # classifiers 是一组用于描述项目属性的字符串。
    #
    # 这些信息通常会被 Python 包管理工具或包索引平台读取，
    # 帮助用户了解项目支持的运行环境和许可证类型。
    classifiers=[

        # 声明该项目使用 Python 3.9。
        "Programming Language :: Python :: 3.9",

        # 声明该项目不依赖某一种特定操作系统。
        #
        # 理论上，它可以在不同操作系统中运行。
        #
        # 但是，实际能否正常运行还取决于：
        # 1. PyTorch；
        # 2. nuPlan；
        # 3. CUDA；
        # 4. 其他第三方依赖；
        # 5. 项目中的系统级配置。
        "Operating System :: OS Independent",

        # 声明该项目允许非商业用途。
        #
        # 该字符串属于项目分类信息。
        #
        # 需要注意：
        # 该描述与后面的 license="MIT" 是否完全一致，
        # 应当结合项目仓库中的 LICENSE 文件进一步确认。
        "License :: Free for non-commercial use",
    ],

    # 设置许可证名称。
    #
    # MIT License 通常是一种较为宽松的开源许可证。
    #
    # 但是，当前代码中的 classifiers 同时包含：
    #
    # "License :: Free for non-commercial use"
    #
    # 两者的表述可能存在差异。
    #
    # 本次仅添加注释，没有改动任何原始代码。
    # 实际使用时，应当以项目仓库中的 LICENSE 文件为准。
    license="MIT",
)
