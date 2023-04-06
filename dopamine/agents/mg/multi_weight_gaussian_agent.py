from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
import gin.tf
import math
import os
import random


@gin.configurable
def piecewise_linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    if step <= warmup_steps:
        return 1.0
    elif step <= (warmup_steps + decay_period / 10.0):
        return epsilon * 10 + (1.0 - epsilon * 10) * (1 - step / (warmup_steps + decay_period / 10.0))
    elif step <= (warmup_steps + decay_period):
        return epsilon + (epsilon * 9) * (1 - step / (warmup_steps + decay_period))
    elif step <= decay_period * 2:
        return 0.005
    elif step <= decay_period * 3:
        return 0.002
    else:
        return 0.0


class BaseGaussianNetwork(keras.Model):
    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128,
                 conv_1: tuple = ([8, 8], 4), conv_2: tuple = ([4, 4], 2), conv_3: tuple = ([3, 3], 1)):
        super(BaseGaussianNetwork, self).__init__(name=name)
        self.action_num = action_num
        self.embedding_dim = embedding_dim

        kernel_initializer = keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
        )
        activation_fn = keras.activations.relu

        self.flatten = keras.layers.Flatten()
        self.conv_1_1 = keras.layers.Conv2D(64, conv_1[0], strides=conv_1[1], kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_1')
        self.conv_1_2 = keras.layers.Conv2D(64, conv_2[0], strides=conv_2[1], kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_2')
        self.conv_1_3 = keras.layers.Conv2D(32, conv_3[0], strides=conv_3[1],  kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_3')

        self.dense_1_1 = keras.layers.Dense(200, kernel_initializer=kernel_initializer, activation=activation_fn,
                                            name='InnerFeature_1')
        self.dense_1_2 = keras.layers.Dense(100, kernel_initializer=kernel_initializer, activation=activation_fn,
                                            name='InnerFeature_2')
        self.dense_1_4 = keras.layers.Dense(embedding_dim, kernel_initializer=kernel_initializer,
                                            activation=None, name='StateFeature_FC')

        self.dense_2_mean_1 = keras.layers.Dense(action_num * 3, kernel_initializer=kernel_initializer,
                                                 activation=activation_fn, name='MeanFeature')
        self.dense_2_mean_2 = keras.layers.Dense(action_num, kernel_initializer=kernel_initializer,
                                                 activation=None, name='Mean')

        self.dense_2_std_1 = keras.layers.Dense(action_num * 3, kernel_initializer=kernel_initializer,
                                                activation=activation_fn, name='STDFeature')
        self.dense_2_std_2 = keras.layers.Dense(action_num, kernel_initializer=kernel_initializer,
                                                activation=None, name='STD')

        self.dense_2_weight_1 = keras.layers.Dense(action_num * 3, kernel_initializer=kernel_initializer,
                                                   activation=activation_fn, name='WeightFeature')
        self.dense_2_weight_2 = keras.layers.Dense(action_num, kernel_initializer=kernel_initializer,
                                                   activation=None, name='Weight')

    def call(self, state: tf.Tensor, is_positive: bool = False) -> tuple:
        """
        前向计算流图
        :param state: tf.Tensor,
            当前状态观测的状态表示张量
        :return: collections.namedtuple
            需要返回的具名元组
        """
        print(f'++++++++++ [{self.name}] 前向计算流图定义 ++++++++++')
        state_shape = state.get_shape().as_list()
        if len(state_shape) == 2:
            print('该环境的状态观测为特征向量形式')
        elif len(state_shape) == 4:
            print('该环境的状态观测为图像矩阵形式')
        else:
            raise ValueError('该环境的状态观测形式存在错误')
        batch_size = state_shape[0]

        # 状态观测 -->> 特征向量 区块
        with tf.compat.v1.name_scope('StateFeaturePatch'):
            if len(state_shape) == 4:
                # 对图像特征进行规范化处理（值域调整为-0.5~0.5）
                # BatchSize x Height x Width x Depth
                state = tf.cast(state, tf.float32) / 255.0 - 0.5
                with tf.compat.v1.name_scope('InnerFeature'):
                    state = self.conv_1_3(self.conv_1_2(self.conv_1_1(state)))
                with tf.compat.v1.name_scope('OuterFeature'):
                    # BatchSize x EMDim
                    state_feature_vec = self.dense_1_4(self.flatten(state))
            else:
                # BatchSize x OriFeatureNum
                state = tf.cast(state, tf.float32)
                with tf.compat.v1.name_scope('InnerFeature'):
                    # BatchSize x 100
                    state = self.dense_1_2(self.dense_1_1(state))
                with tf.compat.v1.name_scope('OuterFeature'):
                    # BatchSize x EMDim
                    state_feature_vec = self.dense_1_4(state)

        # 特征向量 -->> 均值对 区块
        with tf.compat.v1.name_scope('MeanPatch'):
            # BatchSize x ActionNum
            action_mean = self.dense_2_mean_2(self.dense_2_mean_1(state_feature_vec))
            # BatchSize x ActionNum x 1
            action_mean = tf.reshape(action_mean, [batch_size, self.action_num, 1])
            if is_positive:
                action_mean = tf.math.softplus(action_mean + 2)

        # 特征向量 -->> 标准差对 区块
        with tf.compat.v1.name_scope('STDPatch'):
            # BatchSize x ActionNum
            action_std = self.dense_2_std_2(self.dense_2_std_1(state_feature_vec))
            # BatchSize x ActionNum x 1
            action_std = tf.reshape(action_std, [batch_size, self.action_num, 1]) / 2
            print('限制标准差的取值范围')
            action_std = tf.tanh(action_std) * 4
            action_std = tf.exp(action_std)/2

        # 特性向量 -->> 权重 区块
        with tf.compat.v1.name_scope('WeightPatch'):
            # BatchSize x ActionNum
            action_weight = self.dense_2_weight_2(self.dense_2_weight_1(state_feature_vec))
            # BatchSize x ActionNum x 1
            action_weight = tf.reshape(action_weight, [batch_size, self.action_num, 1]) / 2

        return action_mean, action_std, action_weight


@gin.configurable
class TriangleWeightGaussianNetwork(keras.Model):
    TriangleGaussianNetworkType = collections.namedtuple(
        'TriangleGaussianNetwork',
        ['action_mean', 'action_std', 'action_weight']
    )
    GaussianNum = 3

    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128):
        super(TriangleWeightGaussianNetwork, self).__init__(name=name)
        self.action_num = action_num
        self.embedding_dim = embedding_dim

        square_params = [((8, 8), 4), ((4, 4), 2), ((3, 3), 1)]
        horizon_params = [((16, 4), (4, 2)), ((3, 6), 2), ((3, 3), 1)]
        vertical_params = [((4, 16), (2, 4)), ((6, 3), 2), ((3, 3), 1)]
        # square_params = [((6, 6), 3), ((4, 4), 2), ((3, 3), 1)]
        # horizon_params = [((8, 8), 4), ((4, 4), 2), ((3, 3), 1)]
        # vertical_params = [((10, 10), 5), ((4, 4), 2), ((3, 3), 1)]
        self.convnet_1 = BaseGaussianNetwork(action_num, name='SquareConv', conv_1=square_params[0],
                                          conv_2=square_params[1], conv_3=square_params[2])
        self.convnet_2 = BaseGaussianNetwork(action_num, name='HorizonConv', conv_1=horizon_params[0],
                                          conv_2=horizon_params[1], conv_3=horizon_params[2])
        self.convnet_3 = BaseGaussianNetwork(action_num, name='VerticalConv', conv_1=vertical_params[0],
                                          conv_2=vertical_params[1], conv_3=vertical_params[2])

    def call(self, state: tf.Tensor) -> collections.namedtuple:
        """
        前向计算流图
        :param state: tf.Tensor,
            当前状态观测的状态表示张量
        :return: collections.namedtuple
            需要返回的具名元组
        """
        # BatchSize x ActionNum x 1
        action_mean_1, action_std_1, action_weight_1 = self.convnet_1(state, False)
        action_mean_2, action_std_2, action_weight_2 = self.convnet_2(state, False)
        action_mean_3, action_std_3, action_weight_3 = self.convnet_3(state, False)

        # BatchSize x ActionNum x 3
        action_mean = tf.concat([action_mean_1, action_mean_2, action_mean_3], axis=2)
        action_std = tf.concat([action_std_1, action_std_2, action_std_3], axis=2)
        action_weight = tf.concat([action_weight_1, action_weight_2, action_weight_3], axis=2)
        action_weight = tf.nn.softmax(action_weight / 5, axis=2)
        return self.TriangleGaussianNetworkType(
            action_mean=action_mean, action_std=action_std, action_weight=action_weight
        )


@gin.configurable
class FiveWeightGaussianNetwork(keras.Model):
    FiveGaussianNetworkType = collections.namedtuple(
        'FiveGaussianNetwork',
        ['action_mean', 'action_std', 'action_weight']
    )
    GaussianNum = 5

    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128):
        super(FiveWeightGaussianNetwork, self).__init__(name=name)
        self.action_num = action_num
        self.embedding_dim = embedding_dim

        square_params = [((8, 8), 4), ((4, 4), 2), ((3, 3), 1)]
        horizon_params = [((16, 4), (4, 2)), ((3, 6), 2), ((3, 3), 1)]
        vertical_params = [((4, 16), (2, 4)), ((6, 3), 2), ((3, 3), 1)]
        horizon_2_params = [((14, 6), (4, 2)), ((4, 5), 2), ((3, 3), 1)]
        vertical_2_params = [((6, 14), (2, 4)), ((5, 4), 2), ((3, 3), 1)]
        # square_params = [((6, 6), 3), ((4, 4), 2), ((3, 3), 1)]
        # horizon_params = [((8, 8), 4), ((4, 4), 2), ((3, 3), 1)]
        # vertical_params = [((10, 10), 5), ((4, 4), 2), ((3, 3), 1)]
        self.convnet_1 = BaseGaussianNetwork(action_num, name='SquareConv', conv_1=square_params[0],
                                          conv_2=square_params[1], conv_3=square_params[2])
        self.convnet_2 = BaseGaussianNetwork(action_num, name='HorizonConv', conv_1=horizon_params[0],
                                          conv_2=horizon_params[1], conv_3=horizon_params[2])
        self.convnet_3 = BaseGaussianNetwork(action_num, name='VerticalConv', conv_1=vertical_params[0],
                                          conv_2=vertical_params[1], conv_3=vertical_params[2])
        self.convnet_4 = BaseGaussianNetwork(action_num, name='HorizonConv-2', conv_1=horizon_2_params[0],
                                          conv_2=horizon_2_params[1], conv_3=horizon_2_params[2])
        self.convnet_5 = BaseGaussianNetwork(action_num, name='VerticalConv-2', conv_1=vertical_2_params[0],
                                          conv_2=vertical_2_params[1], conv_3=vertical_2_params[2])

    def call(self, state: tf.Tensor) -> collections.namedtuple:
        """
        前向计算流图
        :param state: tf.Tensor,
            当前状态观测的状态表示张量
        :return: collections.namedtuple
            需要返回的具名元组
        """
        # BatchSize x ActionNum x 1
        action_mean_1, action_std_1, action_weight_1 = self.convnet_1(state, False)
        action_mean_2, action_std_2, action_weight_2 = self.convnet_2(state, False)
        action_mean_3, action_std_3, action_weight_3 = self.convnet_3(state, False)
        action_mean_4, action_std_4, action_weight_4 = self.convnet_4(state, False)
        action_mean_5, action_std_5, action_weight_5 = self.convnet_5(state, False)

        # BatchSize x ActionNum x 3
        action_mean = tf.concat([action_mean_1, action_mean_2, action_mean_3, action_mean_4, action_mean_5], axis=2)
        action_std = tf.concat([action_std_1, action_std_2, action_std_3, action_std_4, action_std_5], axis=2)
        action_weight = tf.concat([action_weight_1, action_weight_2, action_weight_3, action_weight_4, action_weight_5],
                                  axis=2)
        action_weight = tf.nn.softmax(action_weight / 5, axis=2)
        return self.FiveGaussianNetworkType(
            action_mean=action_mean, action_std=action_std, action_weight=action_weight
        )


@gin.configurable
class HeptadWeightGaussianNetwork(keras.Model):
    HeptadGaussianNetworkType = collections.namedtuple(
        'HeptadGaussianNetwork',
        ['action_mean', 'action_std', 'action_weight']
    )
    GaussianNum = 7

    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128):
        super(HeptadWeightGaussianNetwork, self).__init__(name=name)
        self.action_num = action_num
        self.embedding_dim = embedding_dim

        square_params = [((8, 8), 4), ((4, 4), 2), ((3, 3), 1)]
        horizon_params = [((16, 4), (4, 2)), ((3, 6), 2), ((3, 3), 1)]
        vertical_params = [((4, 16), (2, 4)), ((6, 3), 2), ((3, 3), 1)]

        horizon_2_params = [((14, 6), (4, 2)), ((4, 5), 2), ((3, 3), 1)]
        vertical_2_params = [((6, 14), (2, 4)), ((5, 4), 2), ((3, 3), 1)]

        horizon_3_params = [((12, 6), (3, 2)), ((4, 4), 2), ((3, 3), 1)]
        vertical_3_params = [((6, 12), (2, 3)), ((4, 4), 2), ((3, 3), 1)]
        # square_params = [((6, 6), 3), ((4, 4), 2), ((3, 3), 1)]
        # horizon_params = [((8, 8), 4), ((4, 4), 2), ((3, 3), 1)]
        # vertical_params = [((10, 10), 5), ((4, 4), 2), ((3, 3), 1)]
        self.convnet_1 = BaseGaussianNetwork(action_num, name='SquareConv', conv_1=square_params[0],
                                          conv_2=square_params[1], conv_3=square_params[2])
        self.convnet_2 = BaseGaussianNetwork(action_num, name='HorizonConv', conv_1=horizon_params[0],
                                          conv_2=horizon_params[1], conv_3=horizon_params[2])
        self.convnet_3 = BaseGaussianNetwork(action_num, name='VerticalConv', conv_1=vertical_params[0],
                                          conv_2=vertical_params[1], conv_3=vertical_params[2])
        self.convnet_4 = BaseGaussianNetwork(action_num, name='HorizonConv-2', conv_1=horizon_2_params[0],
                                          conv_2=horizon_2_params[1], conv_3=horizon_2_params[2])
        self.convnet_5 = BaseGaussianNetwork(action_num, name='VerticalConv-2', conv_1=vertical_2_params[0],
                                          conv_2=vertical_2_params[1], conv_3=vertical_2_params[2])
        self.convnet_6 = BaseGaussianNetwork(action_num, name='HorizonConv-3', conv_1=horizon_3_params[0],
                                             conv_2=horizon_3_params[1], conv_3=horizon_3_params[2])
        self.convnet_7 = BaseGaussianNetwork(action_num, name='VerticalConv-3', conv_1=vertical_3_params[0],
                                             conv_2=vertical_3_params[1], conv_3=vertical_3_params[2])

    def call(self, state: tf.Tensor) -> collections.namedtuple:
        """
        前向计算流图
        :param state: tf.Tensor,
            当前状态观测的状态表示张量
        :return: collections.namedtuple
            需要返回的具名元组
        """
        # BatchSize x ActionNum x 1
        action_mean_1, action_std_1, action_weight_1 = self.convnet_1(state, False)
        action_mean_2, action_std_2, action_weight_2 = self.convnet_2(state, False)
        action_mean_3, action_std_3, action_weight_3 = self.convnet_3(state, False)
        action_mean_4, action_std_4, action_weight_4 = self.convnet_4(state, False)
        action_mean_5, action_std_5, action_weight_5 = self.convnet_5(state, False)
        action_mean_6, action_std_6, action_weight_6 = self.convnet_6(state, False)
        action_mean_7, action_std_7, action_weight_7 = self.convnet_7(state, False)

        # BatchSize x ActionNum x 3
        action_mean = tf.concat([action_mean_1, action_mean_2, action_mean_3, action_mean_4, action_mean_5,
                                 action_mean_6, action_mean_7], axis=2)
        action_std = tf.concat([action_std_1, action_std_2, action_std_3, action_std_4, action_std_5,
                                action_std_6, action_std_7], axis=2)
        action_weight = tf.concat([action_weight_1, action_weight_2, action_weight_3, action_weight_4, action_weight_5,
                                   action_weight_6, action_weight_7], axis=2)
        action_weight = tf.nn.softmax(action_weight / 5, axis=2)
        return self.HeptadGaussianNetworkType(
            action_mean=action_mean, action_std=action_std, action_weight=action_weight
        )


@gin.configurable
class MultiWeightGaussianAgent(rainbow_agent.RainbowAgent):
    def __init__(self, sess, action_num: int, network=TriangleWeightGaussianNetwork, double_dqn: bool = True,
                 sample_num: int = 192, action_mode: str = 'mean', loss_mode: str = 'mse',
                 before_project_mix_samples: bool = True, prob_clib_value: float = 0.004,
                 per_mode: str = 'kl', p_distance: int = 2,
                 summary_writer=None, summary_writing_frequency=500, print_freq=100000):
        """
        :param sess: tf.Session,
            执行运算过程的TF计算资源会话
        :param action_num: int,
            该离散控制问题的可选动作数量
        :param network: tf.Model,
            从状态-动作对映射到高斯分量统计量组的神经网络
        :param double_dqn: bool, optional=True
            是否使用双网络的方式来解决Q学习中存在的过估计问题
        :param sample_num: int, optional=192
            调用采样算子时，从每一个高斯分量中抽取样本的数量
        :param action_mode: str, optional=('mean', 'max', 'min')
            进行决策时使用的决策模式，'mean'代表均值最大化决策，'max'代表最大值最大化决策，'min'代表最小值最大化决策
        :param loss_mode: str, optional=('mse', 'jtd')
            进行误差计算时使用的误差类型，'mse'代表在统计量函数空间中计算统计量之间的均方误差；
            'jtd'代表在分布函数空间中计算两个分布之间的JT散度
        :param before_project_mix_samples: bool, optional=True
            是否在投影之前将新旧分布的表示样本进行混合
        :param prob_clib_value: float, optional=0.004
            对归属于某一个高斯分量的概率值进行截断的最小数值，默认值为0.004，约为3个标准差的距离
        :param per_mode: str, optional=('kl', 'wasserstein', 'mse', 'jt', 'l')
            进行优先经验回放时选择的模式，'kl'对应了两个分布之间的KL散度；'wasserstein'对应两个分布之间的p阶Wasserstein距离；
            'jt'对应两个分布之间的JT(2,2)散度；'mse'对应统计量之间的均方误差；'l'对应了两个分布之间的p阶L距离。
        :param summary_writer: callable,
            汇总信息写入函数
        :param summary_writing_frequency: int,
            汇总信息写入频次
        :param print_freq: int,
            汇总信息打印频次
        """
        self.action_num = action_num
        self.double_dqn = double_dqn
        self.sample_num = sample_num
        self.print_freq = print_freq
        self.action_mode = action_mode
        self.loss_mode = loss_mode
        self.gaussian_num = network.GaussianNum
        self.before_project_mix_samples = before_project_mix_samples
        self.prob_clib_value = prob_clib_value
        self.per_mode = per_mode
        self.p_distance = p_distance
        print('高斯分量的数量：%d' % self.gaussian_num)

        super(MultiWeightGaussianAgent, self).__init__(
            sess=sess, num_actions=action_num, network=network, summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency
        )

    def _create_network(self, name: str) -> tf.keras.Model:
        network = self.network(self.action_num, name=name)
        return network

    def _build_networks(self):
        self.online_convnet = self._create_network('Online')
        self.target_convnet = self._create_network('Target')
        # +++++++++++++++++++++ 决策点 +++++++++++++++++++++++++++++++
        self._action_node_outputs = self.online_convnet(self.state_ph)
        # 当前状态下动作回报的均值对
        # BatchSize x ActionNum x GaussianNum
        self.cur_action_mean = self._action_node_outputs.action_mean
        # 当前状态下动作回报的标准差对
        # BatchSize x ActionNum x GaussianNum
        self.cur_action_std = self._action_node_outputs.action_std
        # 当前状态下动作回报的权重分配
        # BatchSize x ActionNum x GaussianNum
        self.cur_action_weight = self._action_node_outputs.action_weight
        # BatchSize x ActionNum x SampleNum
        self.cur_sample = self._build_samples_op(self.cur_action_mean, self.cur_action_std, self.cur_action_weight)
        # BatchSize x ActionNum
        self.cur_sample_mean = tf.reduce_mean(self.cur_sample, axis=2)
        # BatchSize x ActionNum
        self.cur_sample_dev = tf.reduce_mean((self.cur_sample - self.cur_sample_mean[:, :, None]) ** 2, axis=2)
        self.cur_sample_std = tf.sqrt(self.cur_sample_dev)
        # BatchSize x ActionNum
        self.cur_action_q = tf.reduce_sum(self.cur_action_mean * self.cur_action_weight, axis=2)
        self.cur_action_min = tf.reduce_min(self.cur_action_mean, axis=2)
        self.cur_action_max = tf.reduce_max(self.cur_action_mean, axis=2)
        # 当前状态下的最优动作
        # Scalar
        if self.action_mode == 'mean':
            print('使用均值系数最大化原则进行决策')
            self._q_argmax = tf.argmax(self.cur_action_q, axis=1)[0]
        elif self.action_mode == 'min':
            print('使用风险回避型原则进行决策')
            self._q_argmax = tf.argmax(self.cur_action_min, axis=1)[0]
        elif self.action_mode == 'max':
            print('使用风险追求型原则进行决策')
            self._q_argmax = tf.argmax(self.cur_action_max, axis=1)[0]

        self.info = [
            self.cur_action_mean[0], self.cur_action_std[0], self.cur_action_q[0], self.cur_action_min[0],
            self.cur_action_max[0], self.cur_action_weight[0]
        ]
        # +++++++++++++++++++++ 决策点 +++++++++++++++++++++++++++++++
        # ++++++++++++++++++ 主网络 +++++++++++++++++++++++++
        self._replay_net_outputs = self.online_convnet(self._replay.states)
        # ++++++++++++++++++ 主网络 +++++++++++++++++++++++++
        # ++++++++++++++++++ 目标网络 +++++++++++++++++++++++++
        self._replay_target_network_outputs = self.target_convnet(self._replay.next_states)
        _replay_online_network_outputs = self.online_convnet(self._replay.next_states)
        # 主网络所给出的Q估计
        # BatchSize
        if self.action_mode == 'mean':
            # BatchSize x ActionNum x GaussianNum
            online_action_weight = _replay_online_network_outputs.action_weight
            # BatchSize x 1
            _online_argmax = tf.argmax(
                tf.reduce_sum(_replay_online_network_outputs.action_mean * online_action_weight, axis=2), axis=1
            )
        elif self.action_mode == 'min':
            _online_argmax = tf.argmax(tf.reduce_min(_replay_online_network_outputs.action_mean, axis=2), axis=1)
        elif self.action_mode == 'max':
            _online_argmax = tf.argmax(tf.reduce_max(_replay_online_network_outputs.action_mean, axis=2), axis=1)
        else:
            raise ValueError('ActionMode的值设置错误')
        # 目标网络所给出的Q估计
        # BatchSize
        if self.action_mode == 'mean':
            # BatchSize x ActionNum x GaussianNum
            target_action_weight = self._replay_target_network_outputs.action_weight
            # BatchSize x 1
            _target_argmax = tf.argmax(
                tf.reduce_sum(self._replay_target_network_outputs.action_mean * target_action_weight, axis=2), axis=1
            )
        elif self.action_mode == 'min':
            _target_argmax = tf.argmax(tf.reduce_min(self._replay_target_network_outputs.action_mean, axis=2), axis=1)
        elif self.action_mode == 'max':
            _target_argmax = tf.argmax(tf.reduce_max(self._replay_target_network_outputs.action_mean, axis=2), axis=1)
        else:
            raise ValueError('ActionMode的值设置错误')
        self._replay_target_network_argmax = _target_argmax if not self.double_dqn else _online_argmax
        # ++++++++++++++++++ 目标网络 +++++++++++++++++++++++++

    def _build_samples_op(self, _target_mean, _target_std, _target_weight, sample_num: int = None):
        """
        从一组代表潜在值分布的加权高斯分量中抽取样本

        :param _target_mean: 均值向量
            BatchSize x [ActionNum] x GaussianNum
        :param _target_std: 标准差向量
            BatchSize x [ActionNum] x GaussianNum
        :param _target_weight: 权重向量
            BatchSize x [ActionNum] x GaussianNum
        :return:
        """
        batch_size = self._replay.batch_size
        sample_num = sample_num if sample_num else self.sample_num

        if len(_target_mean.shape) == 2:
            # 1 x SampleNum
            rd_matrix = tf.linspace(start=0.0001, stop=0.9999, num=sample_num)[None, :]
            # BatchSize x GaussianNum
            accu_weight = tf.cumsum(_target_weight, axis=-1)
            dist_mask_list = []
            for k in range(self.gaussian_num):
                init_dist_mask = tf.ones(shape=[batch_size, sample_num])
                for exit_mask in dist_mask_list:
                    init_dist_mask = init_dist_mask * (1 - exit_mask)
                # BatchSize x SampleNum
                dist_mask = init_dist_mask * tf.cast(rd_matrix < accu_weight[:, k:k + 1], tf.float32)
                dist_mask_list.append(dist_mask)

            sample_list = []
            for k in range(self.gaussian_num):
                # BatchSize x SampleNum
                mean = tf.tile(_target_mean[:, k:k + 1], multiples=[1, sample_num])
                # BatchSize x SampleNum
                std = tf.tile(_target_std[:, k:k + 1], multiples=[1, sample_num])
                # BatchSize x SampleNum
                sample = tf.compat.v1.random.normal(shape=[batch_size, sample_num], mean=mean, stddev=std)
                sample_list.append(sample)
        else:
            # 1 x 1 x SampleNum
            rd_matrix = tf.linspace(start=0.0001, stop=0.9999, num=sample_num)[None, None, :]
            # BatchSize x ActionNum x GaussianNum
            accu_weight = tf.cumsum(_target_weight, axis=-1)
            dist_mask_list = []
            for k in range(self.gaussian_num):
                # BatchSize x ActionNum x GaussianNum
                init_dist_mask = tf.ones(shape=[batch_size, self.action_num, sample_num])
                for exit_mask in dist_mask_list:
                    init_dist_mask = init_dist_mask * (1 - exit_mask)
                # BatchSize x ActionNum x SampleNum
                dist_mask = init_dist_mask * tf.cast(rd_matrix < accu_weight[:, :, k:k + 1], tf.float32)
                dist_mask_list.append(dist_mask)

            sample_list = []
            for k in range(self.gaussian_num):
                # BatchSize x ActionNum x SampleNum
                mean = tf.tile(_target_mean[:, :, k:k + 1], multiples=[1, 1, sample_num])
                # BatchSize x ActionNum x SampleNum
                std = tf.tile(_target_std[:, :, k:k + 1], multiples=[1, 1, sample_num])
                # BatchSize x ActionNum x SampleNum
                sample = tf.compat.v1.random.normal(shape=[batch_size, self.action_num, sample_num],
                                                    mean=mean, stddev=std)
                sample_list.append(sample)

        # BatchSize x ... x SampleNum
        sample = sum([d * s for d, s in zip(dist_mask_list, sample_list)])
        return sample

    def _build_target_samples_op(self):
        batch_size = self._replay.batch_size
        # 从数据缓存器中抽取奖励信号和终止信号
        # BatchSize
        rewards = self._replay.rewards
        # BatchSize
        is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
        # BatchSize
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier

        # 抽取最优动作对应的统计量
        # BatchSize x 1
        batch_indices = tf.cast(tf.range(batch_size), tf.int64)[:, None]
        # BatchSize x 2
        gather_indices = tf.concat([batch_indices, self._replay_target_network_argmax[:, None]], axis=1)

        # BatchSize x GaussianNum
        _target_mean = tf.gather_nd(self._replay_target_network_outputs.action_mean, gather_indices)
        # BatchSize x GaussianNum
        _target_std = tf.gather_nd(self._replay_target_network_outputs.action_std, gather_indices)
        # BatchSize x GaussianNum
        _target_weight = tf.gather_nd(self._replay_target_network_outputs.action_weight, gather_indices)

        # BatchSize x SampleNum
        sample = self._build_samples_op(_target_mean, _target_std, _target_weight)

        # 由于终止信号出现对应的那一帧画面，它的值分布重来没有被训练过，所以输出值是比较随意的。而神经网络所输出的
        # 标准差是由指数函数导出的，很容易出现极端大的数值。这就会导致溢出问题。由于这一部分样本会乘以0，所以他们的
        # 具体取值是没有意义的，只不过要避免数值溢出问题

        return rewards[:, None] + tf.math.multiply_no_nan(y=gamma_with_terminal[:, None], x=sample)

    def _build_target_statistic_op(self, _cur_mean, _cur_std, _cur_weight):
        batch_size = self._replay.batch_size
        # BatchSize x SampleNum
        target_samples = self._build_target_samples_op()
        current_samples = self._build_samples_op(_cur_mean, _cur_std, _cur_weight)
        tf.debugging.check_numerics(target_samples, message='Nan值出现在target_samples')
        tf.debugging.check_numerics(current_samples, message='Nan值出现在current_samples')

        if self.before_project_mix_samples:
            print('进行投影前样本混合')
            # BatchSize x SampleNum
            target_samples = tf.concat([target_samples, current_samples], axis=1)
        else:
            print('不进行投影前样本混合')

        def _calculate_w_distance(curr_samples: tf.Tensor, targ_samples: tf.Tensor, p: int = 2):
            sorted_curr_samples = tf.sort(curr_samples, axis=1)
            sorted_targ_samples = tf.sort(targ_samples, axis=1)
            if p <= 4:
                w_distance = tf.reduce_mean(tf.abs(sorted_curr_samples - sorted_targ_samples) ** p, axis=1) ** (1 / p)
            else:
                w_distance = tf.reduce_max(tf.abs(sorted_curr_samples - sorted_targ_samples), axis=1)
            return w_distance

        def _calculate_l_distance(curr_samples: tf.Tensor, targ_samples: tf.Tensor, p: int = 2):
            sample_num = curr_samples.get_shape().as_list()[1]
            concat_samples = tf.concat([curr_samples, targ_samples], axis=1)
            # BatchSize x SampleNum_1 x 1
            sorted_concat_samples = tf.sort(concat_samples, axis=1)[:, :, None]
            # BatchSize x SampleNum_1
            curr_cdf = tf.reduce_sum(
                tf.cast(curr_samples[:, None, :] <= sorted_concat_samples, tf.float32), axis=2
            ) / sample_num
            targ_cdf = tf.reduce_sum(
                tf.cast(targ_samples[:, None, :] <= sorted_concat_samples, tf.float32), axis=2
            ) / sample_num
            if p <= 4:
                l_distance = tf.reduce_mean(tf.abs(curr_cdf - targ_cdf) ** p, axis=1) ** (1 / p)
            else:
                l_distance = tf.reduce_max(tf.abs(curr_cdf - targ_cdf), axis=1)
            return l_distance

        target_sample_num = target_samples.get_shape().as_list()[1]

        current_samples_2 = self._build_samples_op(_cur_mean, _cur_std, _cur_weight, target_sample_num)
        self.w_distance = _calculate_w_distance(current_samples_2, target_samples, self.p_distance)
        self.l_distance = _calculate_l_distance(current_samples_2, target_samples, self.p_distance)

        # E步
        # 每一个样本归属到每一个类别的概率
        # GaussianNum x BatchSize x SampleNum
        dist_mean_list = [tf.tile(_cur_mean[:, i:i+1], multiples=[1, target_sample_num])
                          for i in range(self.gaussian_num)]
        # GaussianNum x BatchSize x SampleNum
        dist_std_list = [tf.tile(_cur_std[:, i:i+1], multiples=[1, target_sample_num])
                         for i in range(self.gaussian_num)]

        pi = tf.constant(np.pi)
        # GaussianNum x BatchSize x SampleNum
        prob_list = [1 / (tf.sqrt(2 * pi) * std) * tf.exp(-(target_samples - mean) ** 2 / (2 * std ** 2))
                     for mean, std in zip(dist_mean_list, dist_std_list)]
        [tf.debugging.check_numerics(g, message='Nan值出现在prob_list中') for g in prob_list]
        clip_value = self.prob_clib_value
        print('概率值的截断系数为：', clip_value)
        prob_list = [tf.clip_by_value(prob, clip_value, 3) for prob in prob_list]

        # GaussianNum x BatchSize x SampleNum
        weighted_prob_list = [prob_list[i] for i in range(self.gaussian_num)]
        # BatchSize x SampleNum
        total_weight = sum(weighted_prob_list)
        # GaussianNum x BatchSize x SampleNum
        weight_list = [weighted_prob / total_weight for weighted_prob in weighted_prob_list]
        [tf.debugging.check_numerics(g, message='Nan值出现在weight_list中') for g in weight_list]

        # 离散化采样
        print('进行离散化采样')
        # GaussianNum x BatchSize x SampleNum
        random_coef_list = [tf.compat.v1.random.uniform(shape=[batch_size, target_sample_num], minval=0, maxval=1)
                       for i in range(self.gaussian_num)]
        weight_list = [tf.cast(weight > rc, tf.float32) for weight, rc in zip(weight_list, random_coef_list)]

        # M步
        # GaussianNum x BatchSize
        weight_sum_list = [tf.clip_by_value(tf.reduce_sum(weight, axis=1), 1, target_sample_num)
                           for weight in weight_list]
        # GaussianNum x BatchSize
        _tar_mean_list = [tf.reduce_sum(weight * target_samples, axis=1) / weight_sum
                          for weight, weight_sum in zip(weight_list, weight_sum_list)]
        # BatchSize x GaussianNum
        _tar_mean = tf.concat([_tar_mean[:, None] for _tar_mean in _tar_mean_list], axis=1)

        # GaussianNum x BatchSize
        _tar_sigma_list = [tf.reduce_sum(weight * (target_samples - _tar_mean[:, None]) ** 2, axis=1) / weight_sum
                           for weight, weight_sum, _tar_mean in zip(weight_list, weight_sum_list, _tar_mean_list)]
        # BatchSize x GaussianNum
        _tar_sigma = tf.concat([_tar_sigma[:, None] for _tar_sigma in _tar_sigma_list], axis=1)
        _tar_std = tf.sqrt(_tar_sigma)

        # GaussianNum x BatchSize
        _tar_weight_list = [weight_sum / target_sample_num for weight_sum in weight_sum_list]
        # BatchSize x GaussianNum
        _tar_weight = tf.concat([_tar_weight[:, None] for _tar_weight in _tar_weight_list], axis=1)

        # BatchSize x GaussianNum
        print('将目标高斯分布进行排序')
        # BatchSize x GaussianNum
        sorted_index = tf.argsort(_tar_mean, axis=1)
        # BatchSize x GaussianNum
        batch_index = tf.tile(tf.range(0, batch_size)[:, None], [1, self.gaussian_num])
        # BatchSize x GaussianNum x 2
        _tar_index = tf.concat([batch_index[:, :, None], sorted_index[:, :, None]], axis=2)

        # BatchSize x GaussianNum
        _tar_mean = tf.gather_nd(_tar_mean, _tar_index)
        _tar_std = tf.gather_nd(_tar_std, _tar_index)
        _tar_weight = tf.gather_nd(_tar_weight, _tar_index)

        return tf.stop_gradient(_tar_mean), tf.stop_gradient(_tar_std), tf.stop_gradient(_tar_weight)

    def _build_train_op(self):
        scope = tf.compat.v1.get_default_graph().get_name_scope()
        params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                             scope=os.path.join(scope, 'Online'))
        mean_net_params, std_net_params, weight_net_params = [], [], []
        for p in params:
            if 'STDPatch' in p.name:
                std_net_params.append(p)
            elif 'WeightPatch' in p.name:
                weight_net_params.append(p)
            else:
                mean_net_params.append(p)

        batch_size = self._replay.batch_size

        # 实际执行动作的筛选索引
        # BatchSize x 1
        indices = tf.range(batch_size)[:, None]
        # BatchSize x 2
        gather_indices = tf.concat([indices, self._replay.actions[:, None]], axis=1)

        # BatchSize x GaussianNum
        _cur_mean = tf.gather_nd(self._replay_net_outputs.action_mean, gather_indices)
        # BatchSize x GaussianNum
        _cur_std = tf.gather_nd(self._replay_net_outputs.action_std, gather_indices)
        # BatchSize x GaussianNum
        _cur_weight = tf.gather_nd(self._replay_net_outputs.action_weight, gather_indices)

        # BatchSize x GaussianNum
        _tar_mean, _tar_std, _tar_weight = self._build_target_statistic_op(_cur_mean, _cur_std, _cur_weight)

        # BatchSize X GaussianNum
        learning_weight = tf.sqrt(_tar_weight * 2)

        # BatchSize
        mean_loss = tf.reduce_sum((_cur_mean - _tar_mean) ** 2 * learning_weight, axis=1)
        # BatchSize
        std_loss = tf.reduce_sum((_cur_std - _tar_std) ** 2 * learning_weight, axis=1)

        # BatchSize
        weight_loss = tf.reduce_sum(tf.square(_cur_weight - _tar_weight), axis=1)
        # TODO: 高斯混合分布下的KL散度距离
        def _calculate_mg_kl(_target_mean, _tar_std, _tar_weight, _cur_mean, _cur_std, _cur_weight):
            _tar_weight = tf.clip_by_value(_tar_weight, 0.001, 1)
            _tar_std = tf.clip_by_value(_tar_std, 0.001, 200)
            _cur_std = tf.clip_by_value(_cur_std, 0.001, 200)
            weight_kl_loss = tf.reduce_sum(_tar_weight * tf.math.log(_tar_weight / _cur_weight), axis=1)
            gaussian_kl_loss = tf.math.log(_cur_std / _tar_std) - 0.5 + \
                               (_tar_std ** 2 + (_tar_mean - _cur_mean) ** 2) / (2 * _cur_std ** 2)
            mg_kl_loss = weight_kl_loss + tf.reduce_sum(_tar_weight * gaussian_kl_loss, axis=1)
            return mg_kl_loss

        # TODO：高斯混合分布下的Jensen-Tsallis距离
        def _calculate_mg_jt(_target_mean, _tar_std, _tar_weight, _cur_mean, _cur_std, _cur_weight):
            cur_distances = []
            for i in range(self.gaussian_num):
                for j in range(self.gaussian_num):
                    # BatchSize
                    w_i, mean_i, std_i = _cur_weight[:, i], _cur_mean[:, i], _cur_std[:, i]
                    w_j, mean_j, std_j = _cur_weight[:, j], _cur_mean[:, j], _cur_std[:, j]
                    prob_ij = tf.compat.v1.distributions.Normal(loc=mean_j, scale=std_i ** 2 + std_j ** 2).prob(mean_i)
                    weighted_prob_ij = w_i * w_j * prob_ij
                    cur_distances.append(weighted_prob_ij[:, None])
            tar_distances = []
            for i in range(self.gaussian_num):
                for j in range(self.gaussian_num):
                    # BatchSize
                    w_i, mean_i, std_i = _tar_weight[:, i], _tar_mean[:, i], _tar_std[:, i]
                    w_j, mean_j, std_j = _tar_weight[:, j], _tar_mean[:, j], _tar_std[:, j]
                    prob_ij = tf.compat.v1.distributions.Normal(loc=mean_j, scale=std_i ** 2 + std_j ** 2).prob(mean_i)
                    weighted_prob_ij = w_i * w_j * prob_ij
                    tar_distances.append(weighted_prob_ij[:, None])
            cur_tar_distances = []
            for i in range(self.gaussian_num):
                for j in range(self.gaussian_num):
                    # BatchSize
                    w_i, mean_i, std_i = _cur_weight[:, i], _cur_mean[:, i], _cur_std[:, i]
                    w_j, mean_j, std_j = _tar_weight[:, j], _tar_mean[:, j], _tar_std[:, j]
                    prob_ij = tf.compat.v1.distributions.Normal(loc=mean_j, scale=std_i ** 2 + std_j ** 2).prob(mean_i)
                    weighted_prob_ij = w_i * w_j * prob_ij
                    cur_tar_distances.append(weighted_prob_ij[:, None])
            # BatchSize
            cur_distance = tf.reduce_sum(tf.concat(cur_distances, axis=1), axis=1)
            tar_distance = tf.reduce_sum(tf.concat(tar_distances, axis=1), axis=1)
            cur_tar_distance = tf.reduce_sum(tf.concat(cur_tar_distances, axis=1), axis=1)
            return cur_distance + tar_distance - 2*cur_tar_distance

        m_tar_weight = _tar_weight / tf.reduce_sum(_tar_weight, axis=1, keepdims=True)
        # KL散度
        mg_kl_loss_1 = _calculate_mg_kl(_tar_mean, _tar_std, m_tar_weight, _cur_mean, _cur_std, _cur_weight)
        mg_kl_loss_2 = _calculate_mg_kl(_cur_mean, _cur_std, _cur_weight, _tar_mean, _tar_std, m_tar_weight)
        mg_kl_loss = (mg_kl_loss_1 + mg_kl_loss_2) / 2
        # JT散度
        mg_jt_loss = _calculate_mg_jt(_tar_mean, _tar_std, m_tar_weight, _cur_mean, _cur_std, _cur_weight)
        mg_jt_loss = tf.sqrt(tf.clip_by_value(mg_jt_loss, 1e-8, 100))

        # +++++++++++++++++++ 优先经验回放权重更新 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self._replay_scheme == 'prioritized':
            if self.per_mode.lower() == 'wasserstein':
                print('使用[分布函数空间]%d阶Wasserstein距离进行优先经验回放' % self.p_distance)
                prioritized_weight = self.w_distance
            elif self.per_mode.lower() == 'l':
                print('使用[分布函数空间]%d阶L距离进行优先经验回放' % self.p_distance)
                prioritized_weight = self.l_distance
            elif self.per_mode.lower() == 'mse':
                print('使用[统计量函数空间]均方误差进行优先经验回放')
                prioritized_weight = mean_loss + std_loss + weight_loss
            elif self.per_mode.lower() == 'kl':
                print('使用[分布函数空间]KL散度进行经验优先选择')
                prioritized_weight = mg_kl_loss_1
                prioritized_weight = tf.clip_by_value(prioritized_weight, 0.000001, 100)
            elif self.per_mode.lower() == 'jt':
                print('使用[分布函数空间]JT散度进行经验优先选择')
                prioritized_weight = tf.clip_by_value(mg_jt_loss, 0.000001, 100)
            update_priorities_op = self._replay.tf_set_priority(self._replay.indices,
                                                                tf.sqrt(prioritized_weight + 1e-5))
        else:
            print('数据缓存器使用均匀回放')
            update_priorities_op = tf.no_op()
        # +++++++++++++++++++ 优先经验回放权重更新 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with tf.control_dependencies([update_priorities_op]):
            if self.loss_mode == 'mse':
                w2_optimizer = tf.compat.v1.train.AdamOptimizer(0.0000625, epsilon=0.00015)
                w2_train_op = w2_optimizer.minimize(tf.reduce_mean(mean_loss + std_loss + weight_loss))
                weight_train_op = tf.no_op()
            elif self.loss_mode == 'jtd':
                jtd_optimizer = tf.compat.v1.train.AdamOptimizer(0.0000625, epsilon=0.00015)
                w2_train_op = jtd_optimizer.minimize(tf.reduce_mean(mg_jt_loss))
                weight_train_op = tf.no_op()

        return [w2_train_op, weight_train_op, tf.no_op(),
                tf.reduce_mean(mean_loss), tf.reduce_mean(std_loss), tf.reduce_mean(weight_loss),
                _cur_mean[0], _cur_std[0], _cur_weight[0], _tar_mean[0], _tar_std[0], _tar_weight[0]]

    def _train_step(self):
        """Runs a single training step.
            Runs a training op if both:
              (1) A minimum number of frames have been added to the replay buffer.
              (2) `training_steps` is a multiple of `update_period`.
            Also, syncs weights from online to target network if training steps is a
            multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.memory.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                _, _, _, mean_loss, std_loss, weight_loss, cur_mean, cur_std, cur_weight, tar_mean, tar_std, tar_weight,\
                    = self._sess.run(self._train_op)
                if self.training_steps % self.print_freq == 0:
                    print()
                    print(f'[{self.training_steps // 1e5}]MeanLoss: {mean_loss: .2f} STDLoss: {std_loss: .2f}  WeightLoss: {weight_loss: .2f}')
                    for c, t, name in \
                            [(cur_mean, tar_mean, 'Mean'), (cur_std, tar_std, 'STD'), (cur_weight, tar_weight, 'Weight')]:
                        s = '    '.join([f'{ic:.2f} -> {it:.2f}' for ic, it in zip(c, t)])
                        print(f'[{self.training_steps // 1e5}]{name}: ' + s)
                    print('==================================================')
                    print()
                if (self.summary_writer is not None and
                        self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary = self._sess.run(self._merged_summaries)
                    self.summary_writer.add_summary(summary, self.training_steps)

            if self.training_steps % self.target_update_period == 0:
                self._sess.run(self._sync_qt_ops)

        self.training_steps += 1