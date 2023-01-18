# Dopamine
<div align="center">
  <img src="https://google.github.io/dopamine/images/dopamine_logo.png"><br><br>
</div>

本项目将在Google开源 [Dopamine项目](https://github.com/google/dopamine) 的基础上，进一步发展值分布型强化学习算法。 该项目相关依赖请参考[Dopamine](https://github.com/google/dopamine) 。

## 新增算法
### 高斯混合算法
在Dopamine框架下我们新增了高斯混合算法（Mixed Weighted Gaussian，MWG）。相关的论文我们会随后提交至线上。
这一算法的核心创新点在于：
- 提出一种全新的值分布型强化学习算法的规范实现框架`SIF-DRL`，这一框架的核心是两个函数空间（**分布函数空间（Distribution Function Space）**和**统计量函数空间（Statistical Function Space）**)和两个转换算子
（**采样算子（Sampling Operator）**和**投影算子（Projection Operator）**），将Bellman更新和神经网络更新区分开。
<div align="center">
  <img src="https://github.com/xiaojianyang820/Dopamine/blob/main/images/SIF-DRL.jpg"><br><br>
</div>
- 在该框架的指导下，设计了一种基于高斯混合模型进行分布表示的DRL算法--`Mixed Weighted Gaussian(MWG)`算法。
这一算法的核心程序文件放在`dopamine/agents/mg`。

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

可以通过如下的样例命令来开启模型训练，相关依赖参考Dopamine项目的具体介绍。
    
    export game_name=asteroids
    nohup python -um dopamine.discrete_domains.train --base_dir tmp/iqn_${game_name} \
        --gin_files dopamine/agents/implicit_quantile/configs/implicit_quantile_${game_name}.gin -> tmp/iqn_${game_name}_output &

预训练好的模型文件可以在该地址下载，下载好的文件放入到`tmp`文件夹中，使用上面的命令可以重新载入神经网络参数，重启相关训练。

Atari游戏的部分测试视频可以在该地址下获取。

