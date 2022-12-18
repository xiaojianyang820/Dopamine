# Dopamine
[Getting Started](#getting-started) |
[Docs][docs] |
[Baseline Results][baselines] |
[Changelist](https://google.github.io/dopamine/docs/changelist)

<div align="center">
  <img src="https://google.github.io/dopamine/images/dopamine_logo.png"><br><br>
</div>

本项目将在Google开源Dopamine项目的基础上，进一步发展值分布型强化学习算法。

## 新增算法
### 高斯混合算法
在Dopamine框架下我们新增了高斯混合算法（Mixed Weighted Gaussian，MWG）。相关的论文我们会随后提交至线上。
此外，根据MWG算法中提出的DRL标准实现框架，在进行投影操作之前，应该将新分布与旧分布的表示样本进行混合。根据这一个主要创新点，
提出了一种对IQN方法的改进方法，同时，在MWG算法中移除这一个改进方法，以对比这一改进所具有的实际效果。可以基于如下的gin配置文件
运行相应的算法实现：
- 标准MWG算法：dopamine/agents/mg/configs/mwg_zaxxon.gin
- 标准IQN算法：dopamine/agents/implicit_quantile/configs/implicit_quantile_asteroids.gin
- 移除了样本混合的MWG算法：dopamine/agents/mg/configs/mwg_no_mix_zaxxon.gin
- 增加了样本混合的IQN算法：dopamine/agents/implicit_quantile/configs/implicit_quantile_mix_asteroids.gin

可以通过如下的样例命令来开启模型训练，相关依赖参考Dopamine项目的具体介绍。
    
    export game_name=asteroids
    nohup python -um dopamine.discrete_domains.train --base_dir tmp/iqn_${game_name} \
        --gin_files dopamine/agents/implicit_quantile/configs/implicit_quantile_${game_name}.gin -> tmp/iqn_${game_name}_output &

Dopamine is a research framework for fast prototyping of reinforcement learning
algorithms. It aims to fill the need for a small, easily grokked codebase in
which users can freely experiment with wild ideas (speculative research).

Our design principles are:

* _Easy experimentation_: Make it easy for new users to run benchmark
                          experiments.
* _Flexible development_: Make it easy for new users to try out research ideas.
* _Compact and reliable_: Provide implementations for a few, battle-tested
                          algorithms.
* _Reproducible_: Facilitate reproducibility in results. In particular, our
                  setup follows the recommendations given by
                  [Machado et al. (2018)][machado].

Dopamine supports the following agents, implemented with jax:

* DQN ([Mnih et al., 2015][dqn])
* C51 ([Bellemare et al., 2017][c51])
* Rainbow ([Hessel et al., 2018][rainbow])
* IQN ([Dabney et al., 2018][iqn])
* SAC ([Haarnoja et al., 2018][sac])

For more information on the available agents, see the [docs](https://google.github.io/dopamine/docs).

Many of these agents also have a tensorflow (legacy) implementation, though
newly added agents are likely to be jax-only.

This is not an official Google product.

## Getting Started


We provide docker containers for using Dopamine.
Instructions can be found [here](https://google.github.io/dopamine/docker/).

Alternatively, Dopamine can be installed from source (preferred) or installed
with pip. For either of these methods, continue reading at prerequisites.

### Prerequisites

Dopamine supports Atari environments and Mujoco environments. Install the
environments you intend to use before you install Dopamine:

**Atari**

1. Install the atari roms following the instructions from
[atari-py](https://github.com/openai/atari-py#roms).
2. `pip install ale-py` (we recommend using a [virtual environment](virtualenv)):
3. `unzip $ROM_DIR/ROMS.zip -d $ROM_DIR && ale-import-roms $ROM_DIR/ROMS`
(replace $ROM_DIR with the directory you extracted the ROMs to).

**Mujoco**

1. Install Mujoco and get a license
[here](https://github.com/openai/mujoco-py#install-mujoco).
2. Run `pip install mujoco-py` (we recommend using a
[virtual environment](virtualenv)).

### Installing from Source


The most common way to use Dopamine is to install it from source and modify
the source code directly:

```
git clone https://github.com/google/dopamine
```

After cloning, install dependencies:

```
pip install -r dopamine/requirements.txt
```

Dopamine supports tensorflow (legacy) and jax (actively maintained) agents.
View the [Tensorflow documentation](https://www.tensorflow.org/install) for
more information on installing tensorflow.

Note: We recommend using a [virtual environment](virtualenv) when working with Dopamine.

### Installing with Pip

Note: We strongly recommend installing from source for most users.

Installing with pip is simple, but Dopamine is designed to be modified
directly. We recommend installing from source for writing your own experiments.

```
pip install dopamine-rl
```

### Running tests

You can test whether the installation was successful by running the following
from the dopamine root directory.

```
export PYTHONPATH=$PYTHONPATH:$PWD
python -m tests.dopamine.atari_init_test
```

## Next Steps

View the [docs][docs] for more information on training agents.

We supply [baselines][baselines] for each Dopamine agent.

We also provide a set of [Colaboratory notebooks](https://github.com/google/dopamine/tree/master/dopamine/colab)
which demonstrate how to use Dopamine.

## References

[Bellemare et al., *The Arcade Learning Environment: An evaluation platform for
general agents*. Journal of Artificial Intelligence Research, 2013.][ale]

[Machado et al., *Revisiting the Arcade Learning Environment: Evaluation
Protocols and Open Problems for General Agents*, Journal of Artificial
Intelligence Research, 2018.][machado]

[Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning*.
Proceedings of the AAAI Conference on Artificial Intelligence, 2018.][rainbow]

[Mnih et al., *Human-level Control through Deep Reinforcement Learning*. Nature,
2015.][dqn]

[Schaul et al., *Prioritized Experience Replay*. Proceedings of the International
Conference on Learning Representations, 2016.][prioritized_replay]

[Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*,
arXiv preprint arXiv:1812.05905, 2018.][sac]

## Giving credit

If you use Dopamine in your work, we ask that you cite our
[white paper][dopamine_paper]. Here is an example BibTeX entry:

```
@article{castro18dopamine,
  author    = {Pablo Samuel Castro and
               Subhodeep Moitra and
               Carles Gelada and
               Saurabh Kumar and
               Marc G. Bellemare},
  title     = {Dopamine: {A} {R}esearch {F}ramework for {D}eep {R}einforcement {L}earning},
  year      = {2018},
  url       = {http://arxiv.org/abs/1812.06110},
  archivePrefix = {arXiv}
}
```



[docs]: https://google.github.io/dopamine/docs/
[baselines]: https://google.github.io/dopamine/baselines
[machado]: https://jair.org/index.php/jair/article/view/11182
[ale]: https://jair.org/index.php/jair/article/view/10819
[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[a3c]: http://proceedings.mlr.press/v48/mniha16.html
[prioritized_replay]: https://arxiv.org/abs/1511.05952
[c51]: http://proceedings.mlr.press/v70/bellemare17a.html
[rainbow]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17204/16680
[iqn]: https://arxiv.org/abs/1806.06923
[sac]: https://arxiv.org/abs/1812.05905
[dopamine_paper]: https://arxiv.org/abs/1812.06110
[vitualenv]: https://docs.python.org/3/library/venv.html#creating-virtual-environments
