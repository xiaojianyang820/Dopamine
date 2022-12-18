# Dopamine
## 概览
这一文档介绍了一些案例来说明如何使用或者拓展Dopamine。每一个模块的说明文档都存放在[API文档](https://github.com/google/dopamine/blob/master/docs/api_docs/python/index.md) 。

### 文件组织形式
Dopamine按照如下的格式进行组织：
- `jax` 包括由jax实现的智能体和网络模型
- `agent` 包括由TensorFlow实现的智能体
- `atari` 包括Atari方面的专用代码，包括执行试验和预处理的代码
- `common` 包括额外的辅助性功能，包括日志和存储节点
- `replay_memory` 包括Dopamine框架所使用的数据缓存器
- `colab` 包含分析试验结果的代码以及相应的jupyter notebook
- `tests` 包括所有的测试文件

### 训练智能体
#### Atari游戏
标准Atari2600试验的入口在`dopamine/discrete_domains/train.py`，为了执行基础的DQN智能体，

```shell
python -um dopamine.discrete_domains.train \
    --base_dir /tmp/dopamine_runs \
    --gin_files dopamine/agents/dqn/configs/dqn.gin
```

默认情况下，这一指令会开启一个长达200万帧的试验。命令行界面会输出最近训练会话的统计结果：

```
    [...]
    I0824 17:13:33.078342 140196395337472 tf_logging.py:115] gamma: 0.990000
    I0824 17:13:33.795608 140196395337472 tf_logging.py:115] Beginning training...
    Steps executed: 5903 Episode length: 1203 Return: -19.
```

为了获取训练过程的细粒度信息，你可以调整文件`dopamine/agents/dqn/configs/dqn.gin`中的试验参数，
具体来说，可以通过减少`Runner.training_steps`和`Runner.evaluation_steps`来减少训练中所需要的总
次数。这一点对于分析日志文件或者存储节点文件比较重要，因为这些文件会在每一个迭代完成后发生。

更一般来说，整个Dopamine都可以通过[gin configuration framework](https://github.com/google/gin-config)
来进行快捷配置。

#### Non-Atari离散环境
我们也提供了样例配置文件去训练Cartpole或者Acrobot上的智能体。比如说，要在Cartpole上面使用默认设置去训练C51，
就可以使用如下的命令：
```shell
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/rainbow/configs/c51_cartpole.gin
```
你也可以使用如下命令在Acrobot上面训练Rainbow，
```shell
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/agents/rainbow/configs/rainbow_acrobot.gin
```

#### 连续控制环境
连续控制智能体的入口文件是`dopamine/continuous_domains/train.py`。你可能需要一个Mujoco的钥匙文件
才能运行如下的案例。在HalfCheetah环境上运行SAC，可以通过如下的语句：
```shell
python -um dopamine.continuous_domains.train \
  --base_dir /tmp/dopamine_runs \
  --gin_files dopamine/jax/agents/sac/configs/sac.gin
```
在默认情况下，这一语句会开启一个长达3200个回合的试验，在每一个回合中有1000个会话轮次。这一条命令会输出
如下的最近一个回合的统计指标：
```
[...]
I0908 17:19:39.618797 1803949 run_experiment.py:446] Starting iteration 0
I0908 17:19:40.592262 1803949 run_experiment.py:405] Average undiscounted return per training episode: -168.19
I0908 17:19:40.592391 1803949 run_experiment.py:407] Average training steps per second: 1027.80
I0908 17:19:45.699378 1803949 run_experiment.py:427] Average undiscounted return per evaluation episode: -279.07
```
如果想要执行不同的环境或者超参数，调整文件`dopamine/jax/agents/sac/configs/sac.gin`中的参数即可。
对于你的试验来说，你可以提供一个新的gin配置文件，或者使用命令行中的`gin_bindings`参数来覆盖已有的配置文件。

如果想要了解gin更多的信息，可以查阅[gin github repo](https://github.com/google/gin-config) 。

#### 配置智能体
整个Dopamine可以通过[gin configuration framework](https://github.com/google/gin-config) 进行快捷配置。
我们为每一个智能体提供很多不同的配置文件。每一个智能体的主配置文件是彼此之间的标准对照，这里面的超参数被设定为智能体的标准性能。
这些配置文件就是
- `dopamine/agents/dqn/configs/dqn.gin`
- `dopamine/agents/rainbow/configs/c51.gin`
- `dopamine/agents/rainbow/configs/rainbow.gin`
- `dopamine/agents/implicit_quantile/configs/implicit_quantile.gin`

这些配置参数具体选择的原因记录在文档[baselines page](https://github.com/google/dopamine/tree/master/baselines/) 。

我们同时提供了论文中使用的配置参数文件，它们是
- `dopamine/agents/dqn/configs/dqn_nature.gin`
- `dopamine/agents/dqn/configs/dqn_icml.gin`
- `dopamine/agents/rainbow/configs/c51_icml.gin`
- `dopamine/agents/implicit_quantile/configs/implicit_quantile_icml.gin`

所有的这些都使用ALE的确定版本，超参数配置上有一些轻微差异。

#### 存储节点和日志
Dopamine提供了基本的功能来运行试验。这一功能可以拆分为两部分：存储节点和日志。这两个功能都依赖于命令行参数`base_dir`，这一个参数通知
Dopamine在哪一个文件夹存储试验结果。

##### 存储节点
默认情况下，Dopamine会在每一次迭代后存储一个试验节点，每一个迭代包括一个训练阶段和一个评估阶段，遵循[Mnih等](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) 
的标准设定。存储节点会保存在`base_dir`下面的`checkpoints`文件夹中。在一个比较高的层次来看，如下的内容会被存储：

- 试验的统计数据（比如迭代次数，学习曲线等等）。这一方面的数据产生于文件`dopamine/atari/run_experiment.py`中的`run_experiment`方法。
- 智能体变量，包括TensorFlow的计算图。这些数据由`dopamine/agents/dqn/dqn_agent.py`的`bundle_and_checkpoint`方法和`unbundle`方法中。
- 数据缓存器中的数据。Atari 2600数据缓存器会占用很大的存储空间，因此，Dopamine使用了额外的代码来保持内存占用处于较低的水平。相关的方法可以
在文件`dopamine/agents/replay_memory/circular_replay_buffer.py`中找到，叫做`save`和`load`方法。

如果你好奇的话，存储节点本身的代码存放在`dopamine/commone/checkpointer.py`。

##### 日志
在每一次迭代的最后，Dopamine同样会记录智能体的效果，包括训练期间以及可选的评估期间。这些日志文件都是通过`dopamine/atari/run_experiment.py`
所生成的，更具体的细节由`dopamine/common/logger.py`进行管理。这些日志文件都是pickle格式的，里面包含了一个字典，以迭代编号（比如`iteration_47`）为
键，而相应的统计指标为值。

读取多个试验日志文件的一种简单方法是使用`colab/utils.py`文件所提供的`read_experiment`方法。

我们提供了一个[colab](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/agents.ipynb) 来说明
如何加载试验中的这些统计结果，然后与我们所提供的基准试验结果进行绘图对比。

#### 修改和拓展智能体
Dopamine的设计目标是使算法研究大大简化。出于这一目标，我们决定去保持一个相对扁平的层级结构，没有设置抽象基类；我们发现这样的设置足以支撑
我们的研究，同时具有便利性和易用性。我们推荐直接修改智能体的代码来满足你的研究需要。

我们提供了一个[colab](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/agents.ipynb) 
来说明如何拓展DQN智能体，或者创建一下新的智能体来进行探索，然后绘制相应的试验结果来与基准实验进行对比。

##### DQN
DQN智能体由两个文件所构成：

- 智能体类，在`dopamine/agents/dqn/dqn_agent.py`
- 数据缓存器类，在`dopamine/replay_memory/circular_replay_buffer.py`

智能体类定义了DQN网络结构，更新法则，以及一些强化学习智能体的基本操作（比如epsilon贪心动作选择，会话存储，回合记录等）。比如，在DQN中使用的
Q学习更新就包括在两个方法中，即`_build_target_q_op`和`_build_train_op`。

##### Rainbow和C51

Rainbow智能体包含在如下的两个文件中：

- 智能体类，在文件`dopamine/agents/rainbow/rainbow_agent.py`中，继承自DQN智能体
- 数据缓存器，在文件`dopamine/replay_memory/prioritized_replay_buffer.py`，继承自DQN智能体的数据缓存器

C51智能体是Rainbow智能体的一种特殊参数配置，具体来说，就是`update_horizon`为1，而采样模式为均匀采样。

##### Implicit Quantile Network（IQN）
IQN智能体定义在另外的一个文件中

- `dopamine/agents/implicit_quantile/implicit_quantile_agent.py`，继承自Rainbow智能体。

#### 下载

我们提供了关于四个智能体在全部60个游戏上的记录文件。这些文件都是`*.tar.gz`格式的文件，需要进行解压缩操作：

- 原始的日志文件可以从[这里](https://storage.cloud.google.com/download-dopamine-rl/compiled_raw_logs_files.tar.gz)获取
    - 你可以查阅这个[colab](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/load_statistics.ipynb) ，来获取如何加载与可视化日志文件的具体指导。
- 编码好的pickle文件可以从[这里](https://storage.cloud.google.com/download-dopamine-rl/compiled_pkl_files.tar.gz) 获得
    - 我们会在智能体和统计结果colabs中使用这一些编码好的pickle文件
- Tensorboard事件文件可以在[这里](https://storage.cloud.google.com/download-dopamine-rl/compiled_tb_event_files.tar.gz) 获得。
    - 我们提供了一个[colab](https://colab.research.google.com/github/google/dopamine/blob/master/dopamine/colab/tensorboard.ipynb) ，
    在这个文件中你可以使用`ngrok`直接启动Tensorboard，在提供的案例中，你的Tensorboard看起来就会是下面这个样子：
    ![tensorboard案例](https://google.github.io/dopamine/images/all_asterix_tb.png)
    
```shell script
*  You can also view these with Tensorboard on your machine. For instance, after
   uncompressing the files you can run:

   
   tensorboard --logdir c51/Asterix/
   

   to display the training runs for C51 on Asterix:
```
!()[https://google.github.io/dopamine/images/c51_asterix_tb.png]

- TensorFlow日志文件包括4个智能体在60个游戏上的5次独立试验。每一个文件的格式为：`https://storage.cloud.google.com/download-dopamine-rl/lucid/${AGENT}/${GAME}/${RUN}/tf_ckpt-199.${SUFFIX}`
，这其中：
    - `AGENT`：可以是`dqn`,`c51`,`rainbow`和`iqn`
    - `GAME`：可以是60个游戏中的任意一个
    - `RUN`：可以是1,2,3,4,5
    - `SUFFIX`：可以是`data-00000-of-00001`，`index`，`meta`其中一个
    
你可以下载全部的这些`.tar.gz`文件，不过这些文件每一个都很大，超过15G。
- [DQN存储文件](https://storage.cloud.google.com/download-dopamine-rl/dqn_checkpoints.tar.gz)
- [C51存储文件](https://storage.cloud.google.com/download-dopamine-rl/c51_checkpoints.tar.gz)
- [Rainbow存储文件](https://storage.cloud.google.com/download-dopamine-rl/rainbow_checkpoints.tar.gz)
- [IQN存储文件](https://storage.cloud.google.com/download-dopamine-rl/iqn_checkpoints.tar.gz)


































