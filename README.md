# Dopamine
<div align="center">
  <img src="https://google.github.io/dopamine/images/dopamine_logo.png"><br><br>
</div>

## English Document
This project will further develop the Distributional Reinforcement Learning(DRL) theory based on the Google open source project [dopamine](https://github.com/google/dopamine). For related dependencies and additional information of this project, please refer to the dopamine project documentation

### New Algorithm
### Mixed Weighted Gaussian Model(MWG)
Under the Dopamine framework, we have added a new model named Mixed Weighted Gaussian. Related papers will be submitted online later.
The core innovation of this algorithm lies in:
- A brand-new specification implementation framework `SIF-DRL` for Distributional Reinforcement Learning algorithms is proposed. The core of this framework is two function spaces (**Distribution Function Space** and **Statistical Function Space**) and two conversion operators
(**Sampling Operator** and **Projection Operator**), distinguish Bellman update from neural network update.
<div align="center">
  <img src="https://github.com/xiaojianyang820/Dopamine/blob/main/images/SIF-DRL.jpg"><br><br>
</div>

- Under the guidance of this framework, a DRL algorithm based on the Gaussian Mixture Model for distribution representation -- `Mixed Weighted Gaussian(MWG)` algorithm -- is designed.
The core program file of this algorithm is placed in `dopamine/agents/mg` .

#### Comparison Test
The core innovation of the algorithm is to mix the original distribution estimate with the new distribution estimate before the projection operation. In order to test the effect of this innovation, two comparative experiments were designed, the comparison between `IQN` without distribution mixing and `Mix_IQN` with distribution mixing, and `No_Mix_MWG` without distribution mixing and `MWG` with distribution mixing contrast.

Their gin configuration files are
- Standard MWG algorithm: `dopamine/agents/mg/configs/mwg_zaxxon.gin`
- Standard IQN algorithm: `dopamine/agents/implicit_quantile/configs/implicit_quantile_asteroids.gin`
- Removed MWG algorithm for sample mixing: `dopamine/agents/mg/configs/mwg_no_mix_zaxxon.gin`
- Added IQN algorithm for sample mixing: `dopamine/agents/implicit_quantile/configs/implicit_quantile_mix_asteroids.gin`

The control effects under these four settings were tested on 12 Atari games as follows,
<table border="2" align="center">
    <th bgcolor="navy"> <td>Mean HNS </td> <td> Median HNS</td> </th>
	<tr >
		<td>IQN</td> <td>3.52</td> <td> 1.26</td>
	</tr>
    <tr >
		<td>Mix_IQN</td> <td>3.83</td> <td> 1.35</td>
	</tr>
    <tr >
		<td>No_Mix_MWG</td> <td>3.66</td> <td> 1.22</td>
	</tr>
    <tr >
		<td>MWG</td> <td>4.57</td> <td> 1.42</td>
	</tr>
</table>
<div align="center">
  <img src="https://github.com/xiaojianyang820/Dopamine/blob/main/images/MixIQN.png"><br><br>
</div>

#### Start Training
You can use the following sample commands to start model training. The `.gin` file is the configuration file of the algorithm parameters.
`base_dir` specifies the storage folder of training logs and model parameters. If the folder is not empty, the program will read the existing parameters first and continue to run the program.
    
    export game_name=asteroids
    python -um dopamine.discrete_domains.train --base_dir tmp/iqn_${game_name} \
        --gin_files dopamine/agents/implicit_quantile/configs/implicit_quantile_${game_name}.gin
	
The pre-trained model file can be downloaded at this [address](https://drive.google.com/drive/folders/1HG2rkYvuisQHmLakWAtRf6J1mY9yRjJ1?usp=sharing), and the downloaded file is placed in the `tmp` folder. Use the above command to reload the neural network parameters and restart the relevant training.

Some test videos of Atari games can be obtained under this [address](https://www.youtube.com/watch?v=HylLIiSdnFA&list=PLLx_dwVwxN9XK36QVFVKTxzqCTXgDF-bE).














## Chinese Document
本项目将在Google开源 [Dopamine项目](https://github.com/google/dopamine) 的基础上，进一步发展值分布型强化学习算法。 该项目相关依赖请参考[Dopamine](https://github.com/google/dopamine) 。

### 新增算法
### 高斯混合算法
在Dopamine框架下我们新增了高斯混合算法（Mixed Weighted Gaussian，MWG）。相关的论文我们会随后提交至线上。
这一算法的核心创新点在于：
- 提出一种全新的值分布型强化学习算法的规范实现框架`SIF-DRL`，这一框架的核心是两个函数空间（**分布函数空间（Distribution Function Space）**和**统计量函数空间（Statistical Function Space）**)和两个转换算子
（**采样算子（Sampling Operator）**和**投影算子（Projection Operator）**），将Bellman更新和神经网络更新区分开。
<div align="center">
  <img src="https://github.com/xiaojianyang820/Dopamine/blob/main/images/SIF-DRL.jpg"><br><br>
</div>

- 在该框架的指导下，设计了一种基于高斯混合模型进行分布表示的DRL算法-- `Mixed Weighted Gaussian(MWG)` 算法。
这一算法的核心程序文件放在 `dopamine/agents/mg` 。

#### 对比试验
该算法的核心创新点在于在投影运算之前将原有分布估计与新分布估计混合到一起。为了检验这一创新点的效应，设计了两个对比试验，无分布混合的`IQN`和有分布混合的`Mix_IQN`的对比，以及无分布混合的`No_Mix_MWG`和有分布混合的`MWG`的对比。
它们的gin配置文件为
- 标准MWG算法：`dopamine/agents/mg/configs/mwg_zaxxon.gin`
- 标准IQN算法：`dopamine/agents/implicit_quantile/configs/implicit_quantile_asteroids.gin`
- 移除了样本混合的MWG算法：`dopamine/agents/mg/configs/mwg_no_mix_zaxxon.gin`
- 增加了样本混合的IQN算法：`dopamine/agents/implicit_quantile/configs/implicit_quantile_mix_asteroids.gin`

在12个Atari游戏上测试了这四种设定下的控制效果如下。
<table border="2" align="center">
    <th bgcolor="navy"> <td>Mean HNS </td> <td> Median HNS</td> </th>
	<tr >
		<td>IQN</td> <td>3.52</td> <td> 1.26</td>
	</tr>
    <tr >
		<td>Mix_IQN</td> <td>3.83</td> <td> 1.35</td>
	</tr>
    <tr >
		<td>No_Mix_MWG</td> <td>3.66</td> <td> 1.22</td>
	</tr>
    <tr >
		<td>MWG</td> <td>4.57</td> <td> 1.42</td>
	</tr>
</table>
<div align="center">
  <img src="https://github.com/xiaojianyang820/Dopamine/blob/main/images/MixIQN.png"><br><br>
</div>

#### 运行算法
可以通过如下的样例命令来开启模型训练，`.gin`文件是算法参数的配置文件，`base_dir`指定了训练日志以及模型参数的存储文件夹，如果该文件夹非空，程序会优先读取其中已有的参数，并继续运行程序。
    
    export game_name=asteroids
    python -um dopamine.discrete_domains.train --base_dir tmp/iqn_${game_name} \
        --gin_files dopamine/agents/implicit_quantile/configs/implicit_quantile_${game_name}.gin

预训练好的模型文件可以在该[地址](https://drive.google.com/drive/folders/1HG2rkYvuisQHmLakWAtRf6J1mY9yRjJ1?usp=sharing)下载，下载好的文件放入到`tmp`文件夹中，使用上面的命令可以重新载入神经网络参数，重启相关训练。

Atari游戏的部分测试视频可以在该[地址](https://www.youtube.com/watch?v=HylLIiSdnFA&list=PLLx_dwVwxN9XK36QVFVKTxzqCTXgDF-bE)下获取。

