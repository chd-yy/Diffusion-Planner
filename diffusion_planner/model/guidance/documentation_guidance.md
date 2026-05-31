<!--
本教程说明如何为 Diffusion Planner 新增一个自定义引导函数。

在扩散模型采样过程中，引导函数会根据当前生成结果 x 计算一个 reward。
后续程序通常会利用 reward 对 x 的梯度，引导采样结果朝着更符合目标的方向变化。

例如，可以设计：
1. 碰撞规避引导；
2. 道路边界约束引导；
3. 舒适性引导；
4. 目标点到达引导；
5. 轨迹平滑性引导。
-->

# Classifier Guidance Tutorial

<!--
标题含义：分类器引导教程。

注意：
这里的原始标题写作 Classifer Guidance Tutorial。
其中 Classifer 通常应拼写为 Classifier。
为了严格保持源代码和原始文本不变，此处不修改原标题。
-->

## Create your own guidance function

<!--
本节介绍如何创建一个自定义引导函数，并将其注册到统一的引导包装器中。

整体流程包括：
1. 创建新的 Python 文件；
2. 定义奖励函数；
3. 将奖励函数添加到 GuidanceWrapper；
4. 运行示例脚本进行验证。
-->

<!--
步骤 1：
在 diffusion_planner/model/guidance/ 目录下创建一个新的 Python 文件。

<my_guidance> 是占位符。
实际使用时，可以替换为具体文件名，例如：
collision_guidance.py
comfort_guidance.py
goal_guidance.py

为了严格保持原始文本不变，下面仍然保留 <my_guidance>。
-->

1. Create ``diffusion_planner/model/guidance/<my_guidance>.py``

```python
# 定义一个自定义引导函数。
#
# 函数名称 my_guidance_fn 可以根据具体任务修改。
# 例如：
# collision_guidance_fn
# comfort_guidance_fn
# goal_guidance_fn
#
# 参数说明：
#
# x：
# 当前扩散采样过程中的轨迹或状态张量。
# 它表示模型当前生成的候选结果。
# 引导函数通常需要根据 x 判断当前轨迹是否满足约束或优化目标。
#
# t：
# 当前扩散时间步。
# 某些引导策略只希望在特定扩散阶段生效，
# 因此可以根据 t 控制奖励函数或梯度是否启用。
#
# cond：
# 条件信息。
# 具体内容取决于模型设计。
# 例如，它可能包含场景编码、地图特征或其他条件变量。
#
# inputs：
# 输入数据。
# 通常可以从中读取邻居车辆状态、地图信息、历史轨迹或掩码。
#
# 返回值：
# reward：
# 用于评价当前生成结果的奖励值。
# 一般而言，reward 越大，表示当前轨迹越符合预期目标。
#
# 注意：
# 当前代码仅展示函数接口。
# 省略号 ... 表示需要自行补充具体的奖励计算逻辑。
def my_guidance_fn(x, t, cond, inputs) -> torch.Tensor:
    ...

    # 返回奖励值。
    #
    # 在引导采样过程中，程序通常会进一步计算：
    #
    # reward 对 x 的梯度
    #
    # 从而确定应该如何调整生成轨迹。
    return reward
```

<!--
步骤 2：
将刚刚创建的引导函数添加到 guidance_wrapper.py 中。

GuidanceWrapper 用于集中管理一个或多个引导函数。
在采样过程中，它可以统一调用这些函数，并将多个奖励组合起来。

<my_guidance_fn> 是占位符。
实际使用时，应替换为步骤 1 中定义的函数名称。

注意：
如果引导函数定义在新的 Python 文件中，
通常还需要在 guidance_wrapper.py 顶部添加对应的 import 语句。
原始教程没有展示 import 部分，因此此处不额外修改原始代码。
-->

2. Add ``<my_guidance_fn>`` in ``diffusion_planner/model/guidance/guidance_wrapper.py``

```python
# 当前代码位于：
#
# diffusion_planner/model/guidance/guidance_wrapper.py
#
# 该文件负责集中管理引导函数。
# diffusion_planner/model/guidance/guidance_wrapper.py

# 省略号表示：
# 此处可能还包含 import 语句、辅助函数或其他已有代码。
...

# 定义引导函数包装器。
#
# GuidanceWrapper 通常用于：
# 1. 保存需要启用的引导函数；
# 2. 在采样阶段依次调用这些引导函数；
# 3. 将多个 reward 合并为一个总奖励；
# 4. 为扩散采样过程提供统一的引导接口。
class GuidanceWrapper:

    # 初始化 GuidanceWrapper。
    def __init__(self):

        # 使用列表保存所有需要启用的引导函数。
        #
        # 列表中的每个元素都是一个可调用函数。
        #
        # 每个函数通常具有类似接口：
        #
        # guidance_fn(x, t, cond, inputs) -> reward
        #
        # <my_guidance_1>、<my_guidance_2> 和 <my_guidance_N>
        # 均为占位符。
        #
        # 实际使用时，可以替换为真实函数，例如：
        #
        # collision_guidance_fn
        # comfort_guidance_fn
        # goal_guidance_fn
        #
        # 注意：
        # 尖括号形式的占位符不是有效的 Python 语法。
        # 它们只是用于说明应该在此处填入自定义函数。
        self._guidance_fns = [
            <my_guidance_1>,
            <my_guidance_2>,
            ...
            <my_guidance_N>
        ]

    # 定义 GuidanceWrapper 实例被调用时的行为。
    #
    # __call__ 方法使得 GuidanceWrapper 的实例可以像函数一样调用。
    #
    # 例如：
    #
    # wrapper = GuidanceWrapper()
    # reward = wrapper(...)
    #
    # 省略号表示：
    # 具体参数和内部实现未在当前教程中展开。
    def __call__(...):

        # 省略号表示：
        # 此处通常会遍历 self._guidance_fns，
        # 调用每一个引导函数，并组合对应的奖励值。
        ...
        
# 省略号表示：
# 文件中可能还包含其他代码。
...
```

<!--
步骤 3：
运行 sim_guidance_demo.sh 脚本。

该脚本通常用于启动带有 guidance 的仿真示例，
以验证新增引导函数是否能够正常执行。

运行前应确保：
1. 新的引导函数已经创建；
2. guidance_wrapper.py 已经导入该函数；
3. 引导函数已经添加到 self._guidance_fns 列表；
4. Python 环境和项目依赖已经正确安装。
-->

3. Run ``sim_guidance_demo.sh``

<!--
步骤 4：
如果脚本能够正常运行，就可以观察自定义 guidance 对轨迹生成结果的影响。

例如：
1. 碰撞次数是否减少；
2. 轨迹是否更加平滑；
3. 自车是否更倾向于满足道路约束；
4. 引导强度是否过大或过小；
5. 是否出现梯度不稳定问题。
-->

4. Enjoy.