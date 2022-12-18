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
import tensorflow_probability as tfp


@gin.configurable
class Double_Gaussian_Network(keras.Model):
    Double_Gaussian_NetworkType = collections.namedtuple(
        'Double_Gaussian_Network',
        ['action_mean', 'action_std']
    )

    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128):
        super(Double_Gaussian_Network, self).__init__(name=name)

        self.action_num = action_num
        self.embedding_dim = embedding_dim

        kernel_initializer = keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
        )
        activation_fn = keras.activations.relu

        self.flatten = keras.layers.Flatten()
        self.conv_1_1 = keras.layers.Conv2D(128, [8, 8], strides=4, kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_1')
        self.conv_1_2 = keras.layers.Conv2D(128, [4, 4], strides=2, kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_2')
        self.conv_1_3 = keras.layers.Conv2D(64, [3, 3], strides=1, kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_3')

        self.dense_1_1 = keras.layers.Dense(200, kernel_initializer=kernel_initializer, activation=activation_fn,
                                            name='InnerFeature_1')
        self.dense_1_2 = keras.layers.Dense(100, kernel_initializer=kernel_initializer, activation=activation_fn,
                                            name='InnerFeature_2')
        self.dense_1_4 = keras.layers.Dense(embedding_dim, kernel_initializer=kernel_initializer,
                                            activation=None, name='StateFeature_FC')

        self.dense_2_mean_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                               activation=activation_fn, name='MeanFeature')
        self.dense_2_mean_2 = keras.layers.Dense(action_num * 2, kernel_initializer=kernel_initializer,
                                                 activation=None, name='MeanFeature')

        self.dense_2_std_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                                 activation=activation_fn, name='STDFeature')
        self.dense_2_std_2 = keras.layers.Dense(action_num * 2, kernel_initializer=kernel_initializer,
                                                 activation=None, name='STDFeature')

    def call(self, state: tf.Tensor) -> collections.namedtuple:
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
            # BatchSize x [ActionNum * 2]
            action_mean = self.dense_2_mean_2(self.dense_2_mean_1(state_feature_vec))
            # BatchSize x ActionNum x 2
            action_mean = tf.reshape(action_mean, [batch_size, self.action_num, 2]) + tf.constant([-1.0, 1.0])

        # 特征向量 -->> 对数标准差对 区块
        with tf.compat.v1.name_scope('STDPatch'):
            # BatchSize x [ActionNum * 2]
            action_std = self.dense_2_std_2(self.dense_2_std_1(state_feature_vec))
            # BatchSize x ActionNum x 2
            action_std = tf.reshape(action_std, [batch_size, self.action_num, 2]) / 5
            action_std = tf.exp(action_std)

        return self.Double_Gaussian_NetworkType(
            action_mean=action_mean, action_std=action_std
        )


@gin.configurable
class Double_Gaussian_V2_Network(keras.Model):
    Double_Gaussian_NetworkType = collections.namedtuple(
        'Double_Gaussian_Network',
        ['action_mean', 'action_std']
    )

    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128):
        super(Double_Gaussian_V2_Network, self).__init__(name=name)
        print('使用V2版本的网络结构')
        self.action_num = action_num
        self.embedding_dim = embedding_dim

        kernel_initializer = keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
        )
        activation_fn = keras.activations.relu

        self.flatten = keras.layers.Flatten()
        self.conv_1_1 = keras.layers.Conv2D(128, [8, 8], strides=4, kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_1')
        self.conv_1_2 = keras.layers.Conv2D(128, [4, 4], strides=2, kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_2')
        self.conv_1_3 = keras.layers.Conv2D(64, [3, 3], strides=1, kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='Conv_3')

        self.dense_1_1 = keras.layers.Dense(200, kernel_initializer=kernel_initializer, activation=activation_fn,
                                            name='InnerFeature_1')
        self.dense_1_2 = keras.layers.Dense(100, kernel_initializer=kernel_initializer, activation=activation_fn,
                                            name='InnerFeature_2')
        self.dense_1_4 = keras.layers.Dense(embedding_dim, kernel_initializer=kernel_initializer,
                                            activation=None, name='StateFeature_FC')

        self.dense_2_mean_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                               activation=activation_fn, name='MeanFeature')
        self.dense_2_mean_2 = keras.layers.Dense(action_num * 1, kernel_initializer=kernel_initializer,
                                                 activation=None, name='MeanFeature')

        self.dense_2_std_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                                 activation=activation_fn, name='STDFeature')
        self.dense_2_std_2 = keras.layers.Dense(action_num * 1, kernel_initializer=kernel_initializer,
                                                 activation=None, name='STDFeature')
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.s_conv_1_1_h = keras.layers.Conv2D(128, [16, 4], strides=[4, 2], kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='S_Conv_1_H')
        self.s_conv_1_2_h = keras.layers.Conv2D(128, [3, 12], strides=[3, 6], kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='S_Conv_2_H')
        self.s_conv_1_3_h = keras.layers.Conv2D(64, [3, 3], strides=1, kernel_initializer=kernel_initializer,
                                                activation=activation_fn, padding='same', name='S_Conv_3_H')

        self.s_conv_1_1_v = keras.layers.Conv2D(128, [4, 16], strides=[2, 4], kernel_initializer=kernel_initializer,
                                              activation=activation_fn, padding='same', name='S_Conv_1_V')
        self.s_conv_1_2_v = keras.layers.Conv2D(128, [12, 3], strides=[6, 3], kernel_initializer=kernel_initializer,
                                              activation=activation_fn, padding='same', name='S_Conv_2_V')
        self.s_conv_1_3_v = keras.layers.Conv2D(64, [3, 3], strides=1, kernel_initializer=kernel_initializer,
                                            activation=activation_fn, padding='same', name='S_Conv_3_V')

        self.s_dense_1_4 = keras.layers.Dense(embedding_dim, kernel_initializer=kernel_initializer,
                                            activation=None, name='S_StateFeature_FC')

        self.s_dense_2_mean_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                                 activation=activation_fn, name='S_MeanFeature')
        self.s_dense_2_mean_2 = keras.layers.Dense(action_num * 1, kernel_initializer=kernel_initializer,
                                                 activation=None, name='S_MeanFeature')

        self.s_dense_2_std_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                                activation=activation_fn, name='S_STDFeature')
        self.s_dense_2_std_2 = keras.layers.Dense(action_num * 1, kernel_initializer=kernel_initializer,
                                                activation=None, name='S_STDFeature')

    def call(self, state: tf.Tensor) -> collections.namedtuple:
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
                    state_h = self.s_conv_1_3_h(self.s_conv_1_2_h(self.s_conv_1_1_h(state)))
                    state_v = self.s_conv_1_3_v(self.s_conv_1_2_v(self.s_conv_1_1_v(state)))
                with tf.compat.v1.name_scope('OuterFeature'):
                    # BatchSize x EMDim
                    state_feature_vec = self.dense_1_4(self.flatten(state))
                    state_hv_feature_vec = self.s_dense_1_4(self.flatten(state_h + state_v))
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
            # BatchSize x [ActionNum * 1]
            action_mean_cache = self.dense_2_mean_2(self.dense_2_mean_1(state_feature_vec))
            # BatchSize x ActionNum x 1
            action_mean = tf.reshape(action_mean_cache, [batch_size, self.action_num])[:, :, None]

            # BatchSize x [ActionNum * 1]
            action_extra_mean = self.s_dense_2_mean_2(self.s_dense_2_mean_1(state_hv_feature_vec))
            '''
            action_extra_mean = tf.math.softplus(action_extra_mean + 2)
            # BatchSize x ActionNum x 1
            action_hv_mean = tf.reshape(action_extra_mean + action_mean_cache, [batch_size, self.action_num])[:, :, None]
            '''
            action_hv_mean = tf.reshape(action_extra_mean, [batch_size, self.action_num])[:, :, None]

        # 特征向量 -->> 对数标准差对 区块
        with tf.compat.v1.name_scope('STDPatch'):
            # BatchSize x [ActionNum * 1]
            action_std = self.dense_2_std_2(self.dense_2_std_1(state_feature_vec))
            # BatchSize x ActionNum x 1
            action_std = tf.reshape(action_std, [batch_size, self.action_num])[:, :, None] / 2
            action_std = tf.exp(action_std)
            #action_std = tf.math.softplus(action_std + 1)

            # BatchSize x [ActionNum * 1]
            action_hv_std = self.s_dense_2_std_2(self.s_dense_2_std_1(state_hv_feature_vec))
            # BatchSize x ActionNum x 1
            action_hv_std = tf.reshape(action_hv_std, [batch_size, self.action_num])[:, :, None] / 2
            action_hv_std = tf.exp(action_hv_std)
            #action_hv_std = tf.math.softplus(action_hv_std + 1)

        # BatchSize x ActionNum x 2
        action_mean = tf.concat([action_mean, action_hv_mean], axis=2)
        action_std = tf.concat([action_std, action_hv_std], axis=2)

        return self.Double_Gaussian_NetworkType(
            action_mean=action_mean, action_std=action_std
        )


class BaseConvResNetwork(keras.Model):
    def __init__(self, name: str = None, conv_kernel=[7, 7]):
        super(BaseConvResNetwork, self).__init__(name=name)

        kernel_initializer = keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
        )

        self.flatten = keras.layers.Flatten()
        self.conv_1_1 = keras.layers.Conv2D(32, conv_kernel, strides=1, kernel_initializer=kernel_initializer,
                                            activation=None, padding='same', name='Conv_1_1')
        self.conv_1_2 = keras.layers.MaxPool2D([3, 3], strides=2, name='Conv_1_2')
        self.conv_1_3_1 = keras.layers.Conv2D(32, [3, 3], strides=1, kernel_initializer=kernel_initializer,
                                              activation=None, padding='same', name='Conv_1_3_1')
        self.conv_1_3_2 = keras.layers.Conv2D(32, [3, 3], strides=1, kernel_initializer=kernel_initializer,
                                              activation=None, padding='same', name='Conv_1_3_2')
        self.conv_1_4_1 = keras.layers.Conv2D(32, [3, 3], strides=1, kernel_initializer=kernel_initializer,
                                              activation=None, padding='same', name='Conv_1_4_1')
        self.conv_1_4_2 = keras.layers.Conv2D(32, [3, 3], strides=1, kernel_initializer=kernel_initializer,
                                              activation=None, padding='same', name='Conv_1_4_2')

    def call(self, state):
        statefeature_1 = self.conv_1_2(self.conv_1_1(state))

        statefeature_2 = tf.nn.relu(statefeature_1)
        statefeature_3 = tf.nn.relu(self.conv_1_3_1(statefeature_2))
        statefeature_4 = self.conv_1_3_2(statefeature_3)
        statefeature_5 = statefeature_4 + statefeature_1

        statefeature_6 = tf.nn.relu(statefeature_5)
        statefeature_7 = tf.nn.relu(self.conv_1_4_1(statefeature_6))
        statefeature_8 = self.conv_1_4_2(statefeature_7)
        statefeature_9 = statefeature_8 + statefeature_5

        return statefeature_9


@gin.configurable
class GaussianResNetwork(keras.Model):
    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128, conv_kernel=[7, 7]):
        super(GaussianResNetwork, self).__init__(name=name)
        kernel_initializer = keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
        )
        activation_fn = keras.activations.relu

        self.action_num = action_num
        self.embedding_dim = embedding_dim

        self.res_1 = BaseConvResNetwork('Res_1', conv_kernel)
        self.res_2 = BaseConvResNetwork('Res_2', conv_kernel)
        self.res_3 = BaseConvResNetwork('Res_3', conv_kernel)
        self.flatten = keras.layers.Flatten()
        self.dense_1_mean_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                                 activation=activation_fn, name='MeanFeature')
        self.dense_1_mean_2 = keras.layers.Dense(action_num * 1, kernel_initializer=kernel_initializer,
                                                 activation=None, name='Mean')
        self.dense_2_mean_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                                 activation=activation_fn, name='StdFeature')
        self.dense_2_mean_2 = keras.layers.Dense(action_num * 1, kernel_initializer=kernel_initializer,
                                                 activation=None, name='Std')

    def call(self, state):
        state_shape = state.get_shape().as_list()
        batch_size = state_shape[0]
        # 对图像特征进行规范化处理（值域调整为-0.5~0.5）
        # BatchSize x Height x Width x Depth
        state = tf.cast(state, tf.float32) / 255.0 - 0.5
        res_feature = self.res_3(self.res_2(self.res_1(state)))
        res_feature = self.flatten(tf.nn.relu(res_feature))
        # BatchSize x [ActionNum * 1]
        mean = self.dense_1_mean_2(self.dense_1_mean_1(res_feature))
        # BatchSize x ActionNum x 1
        action_mean = tf.reshape(mean, [batch_size, self.action_num, 1])

        # BatchSize x [ActionNum * 1]
        std = self.dense_2_mean_2(self.dense_2_mean_1(res_feature))
        # BatchSize x ActionNum x 1
        action_std = tf.reshape(std, [batch_size, self.action_num, 1]) / 2
        action_std = tf.exp(action_std)

        return action_mean, action_std


@gin.configurable
class DoubleGaussianResNetwork(keras.Model):
    DoubleGaussianNetworkType = collections.namedtuple(
        'DoubleGaussianNetwork',
        ['action_mean', 'action_std']
    )
    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128):
        super(DoubleGaussianResNetwork, self).__init__(name=name)
        self.gaussian_net_1 = GaussianResNetwork(action_num, 'GaussianNet-1', conv_kernel=[9, 5])
        self.gaussian_net_2 = GaussianResNetwork(action_num, 'GaussianNet-2', conv_kernel=[5, 9])

    def call(self, state):
        action_mean_1, action_std_1 = self.gaussian_net_1(state)
        action_mean_2, action_std_2 = self.gaussian_net_2(state)
        action_mean = tf.concat([action_mean_1, action_mean_2], axis=2)
        action_std = tf.concat([action_std_1, action_std_2], axis=2)
        return self.DoubleGaussianNetworkType(
            action_mean=action_mean, action_std=action_std
        )


@gin.configurable
class DoubleGaussianAgent(rainbow_agent.RainbowAgent):
    def __init__(self, sess, action_num, network=Double_Gaussian_V2_Network, double_dqn: bool = True,
                 sample_num: int = 128, action_mode: str = 'mean', approx_std_weight: float = 0.1,
                 loss_mode: str = 'mse',
                 summary_writer=None, summary_writing_frequency=500, print_freq=100000):
        self.action_num = action_num
        self.double_dqn = double_dqn
        self.sample_num = sample_num
        self.print_freq = print_freq
        self.action_mode = action_mode
        self.approx_std_weight = approx_std_weight
        self.loss_mode = loss_mode

        super(DoubleGaussianAgent, self).__init__(
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
        # BatchSize x ActionNum x 2
        self.cur_action_mean = self._action_node_outputs.action_mean
        # BatchSize x ActionNum
        self.cur_action_diff = tf.abs(self.cur_action_mean[:, :, 0] - self.cur_action_mean[:, :, 1])
        # 当前状态下动作回报的标准差对
        # BatchSize x ActionNum x 2
        self.cur_action_std = self._action_node_outputs.action_std
        # BatchSize x ActionNum x SampleNum
        self.cur_sample = self._build_samples_op(self.cur_action_mean, self.cur_action_std)
        # BatchSize x ActionNum
        self.cur_sample_mean = tf.reduce_mean(self.cur_sample, axis=2)
        # BatchSize x ActionNum
        self.cur_sample_dev = tf.reduce_mean((self.cur_sample - self.cur_sample_mean[:, :, None]) ** 2, axis=2)
        self.cur_sample_std = tf.sqrt(self.cur_sample_dev)
        # BatchSize x ActionNum
        self.cur_appro_neg_q = tf.reduce_mean(self.cur_action_mean -
                                              self.approx_std_weight * self.cur_action_std, axis=2) -\
                               self.approx_std_weight * self.cur_action_diff
        self.cur_appro_pos_q = tf.reduce_mean(self.cur_action_mean +
                                              self.approx_std_weight * self.cur_action_std, axis=2) +\
                               self.approx_std_weight * self.cur_action_diff
        # 当前状态下动作回报的数学期望
        # BatchSize x ActionNum
        self.cur_action_q = tf.reduce_mean(self.cur_action_mean, axis=2)
        self.cur_action_min = tf.reduce_min(self.cur_action_mean, axis=2)
        self.cur_action_max = tf.reduce_max(self.cur_action_mean, axis=2)
        #
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
        elif self.action_mode == 'neg':
            print('使用近似风险回避原则进行决策')
            self._q_argmax = tf.argmax(self.cur_appro_neg_q, axis=1)[0]
        elif self.action_mode == 'pos':
            print('使用近似风险追求原则进行决策')
            self._q_argmax = tf.argmax(self.cur_appro_pos_q, axis=1)[0]

        self.info = [
            self.cur_action_mean[0], self.cur_action_std[0], self.cur_action_q[0], self.cur_action_min[0],
            self.cur_action_max[0], self.cur_appro_neg_q[0], self.cur_appro_pos_q[0]
        ]
        # +++++++++++++++++++++ 决策点 +++++++++++++++++++++++++++++++
        # ++++++++++++++++++ 主网络 +++++++++++++++++++++++++
        self._replay_net_outputs = self.online_convnet(self._replay.states)
        # ++++++++++++++++++ 主网络 +++++++++++++++++++++++++
        # ++++++++++++++++++ 目标网络 +++++++++++++++++++++++++
        self._replay_target_network_outputs = self.target_convnet(self._replay.next_states)
        _replay_online_network_outputs = self.online_convnet(self._replay.next_states)
        # 主网络所给出的Q估计
        _online_action_mean = _replay_online_network_outputs.action_mean
        _online_action_std = _replay_online_network_outputs.action_std
        _online_action_diff = tf.abs(_online_action_mean[:, :, 0] - _online_action_mean[:, :, 1])
        # BatchSize
        if self.action_mode == 'mean':
            _online_argmax = tf.argmax(tf.reduce_mean(_online_action_mean, axis=2), axis=1)
        elif self.action_mode == 'min':
            _online_argmax = tf.argmax(tf.reduce_min(_online_action_mean, axis=2), axis=1)
        elif self.action_mode == 'max':
            _online_argmax = tf.argmax(tf.reduce_max(_online_action_mean, axis=2), axis=1)
        elif self.action_mode == 'neg':
            _online_argmax = tf.argmax(
                tf.reduce_mean(_online_action_mean - self.approx_std_weight * _online_action_std, axis=2) -\
                _online_action_diff * self.approx_std_weight, axis=1
            )
        elif self.action_mode == 'pos':
            _online_argmax = tf.argmax(
                tf.reduce_mean(_online_action_mean + self.approx_std_weight * _online_action_std, axis=2) +\
                _online_action_diff * self.approx_std_weight, axis=1
            )
        else:
            raise ValueError('ActionMode的值设置错误')
        # 目标网络所给出的Q估计
        _target_action_mean = self._replay_target_network_outputs.action_mean
        _target_action_std = self._replay_target_network_outputs.action_std
        _target_action_diff = tf.abs(_target_action_mean[:, :, 0] - _target_action_mean[:, :, 1])
        # BatchSize
        if self.action_mode == 'mean':
            _target_argmax = tf.argmax(tf.reduce_mean(self._replay_target_network_outputs.action_mean, axis=2), axis=1)
        elif self.action_mode == 'min':
            _target_argmax = tf.argmax(tf.reduce_min(self._replay_target_network_outputs.action_mean, axis=2), axis=1)
        elif self.action_mode == 'max':
            _target_argmax = tf.argmax(tf.reduce_max(self._replay_target_network_outputs.action_mean, axis=2), axis=1)
        elif self.action_mode == 'neg':
            _target_argmax = tf.argmax(
                tf.reduce_mean(_target_action_mean - self.approx_std_weight * _target_action_std, axis=2) -\
                _target_action_diff * self.approx_std_weight, axis=1
            )
        elif self.action_mode == 'pos':
            _target_argmax = tf.argmax(
                tf.reduce_mean(_target_action_mean + self.approx_std_weight * _target_action_std, axis=2) +\
                _target_action_diff * self.approx_std_weight, axis=1
            )
        else:
            raise ValueError('ActionMode的值设置错误')
        self._replay_target_network_argmax = _target_argmax if not self.double_dqn else _online_argmax
        # ++++++++++++++++++ 目标网络 +++++++++++++++++++++++++

    def _build_samples_op(self, _target_mean, _target_std, _target_sample_num=None):
        batch_size = self._replay.batch_size
        half_sample_num = _target_sample_num // 2 if _target_sample_num else self.sample_num // 2
        if len(_target_mean.shape) == 2:
            # BatchSize x [SampleNum // 2]
            mean_1 = tf.tile(_target_mean[:, 0:1], multiples=[1, half_sample_num])
            # BatchSize x [SampleNum // 2]
            mean_2 = tf.tile(_target_mean[:, 1:2], multiples=[1, half_sample_num])
            # BatchSize x [SampleNum // 2]
            std_1 = tf.tile(_target_std[:, 0:1], multiples=[1, half_sample_num])
            # BatchSize x [SampleNum // 2]
            std_2 = tf.tile(_target_std[:, 1:2], multiples=[1, half_sample_num])
            # BatchSize x [SampleNum // 2]
            sample_1 = tf.compat.v1.random.normal(shape=[batch_size, half_sample_num], mean=mean_1, stddev=std_1)
            sample_2 = tf.compat.v1.random.normal(shape=[batch_size, half_sample_num], mean=mean_2, stddev=std_2)
        else:
            # BatchSize x ActionNum x [SampleNum // 2]
            mean_1 = tf.tile(_target_mean[:, :, 0:1], multiples=[1, 1, half_sample_num])
            # BatchSize x ActionNum x [SampleNum // 2]
            mean_2 = tf.tile(_target_mean[:, :, 1:2], multiples=[1, 1, half_sample_num])
            # BatchSize x ActionNum x [SampleNum // 2]
            std_1 = tf.tile(_target_std[:, :, 0:1], multiples=[1, 1, half_sample_num])
            # BatchSize x ActionNum x [SampleNum // 2]
            std_2 = tf.tile(_target_std[:, :, 1:2], multiples=[1, 1, half_sample_num])
            # BatchSize x ActionNum x [SampleNum // 2]
            sample_1 = tf.compat.v1.random.normal(shape=[batch_size, self.action_num, half_sample_num], mean=mean_1, stddev=std_1)
            sample_2 = tf.compat.v1.random.normal(shape=[batch_size, self.action_num, half_sample_num], mean=mean_2, stddev=std_2)

        # BatchSize x ... x SampleNum
        sample = tf.concat([sample_1, sample_2], axis=-1)
        return sample

    def _build_target_samples_op(self, _cur_mean, _cur_std):
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

        # BatchSize x 2
        _target_mean = tf.gather_nd(self._replay_target_network_outputs.action_mean, gather_indices)
        # BatchSize x 2
        _target_std = tf.gather_nd(self._replay_target_network_outputs.action_std, gather_indices)

        sample = self._build_samples_op(_target_mean, _target_std)
        #'''
        print('混入当前分布的一部分样本')
        # TODO：在目标样本中混入一部分当前分布下对应的样本
        # 混入当前分布的样本
        # BatchSize x [SampleNum // 2]
        current_samples = self._build_samples_op(_cur_mean, _cur_std, self.sample_num // 2 - 32)
        # BatchSize x [SampleNum // 2]
        reward_samples = tf.tile(rewards[:, None], multiples=[1, self.sample_num // 2 + 32])
        # BatchSize x SampleNum
        ter_target_samples = tf.concat([reward_samples, current_samples], axis=1)
        # BatchSize x SampleNum
        noter_target_samples = rewards[:, None] + gamma_with_terminal[:, None] * sample

        target_samples = is_terminal_multiplier[:, None] * noter_target_samples +\
                           (1 - is_terminal_multiplier[:, None]) * ter_target_samples
        #'''
        #target_samples = rewards[:, None] + gamma_with_terminal[:, None] * sample

        return target_samples

    def _build_target_statistic_op(self, _cur_mean, _cur_std):
        batch_size = self._replay.batch_size
        # 目标样本
        # BatchSize x SampleNum
        target_samples = self._build_target_samples_op(_cur_mean, _cur_std)
        target_sample_num = target_samples.get_shape().as_list()[1]

        # BatchSize x SampleNum
        dist_1_mean = tf.tile(_cur_mean[:, 0:1], multiples=[1, target_sample_num])
        dist_2_mean = tf.tile(_cur_mean[:, 1:2], multiples=[1, target_sample_num])
        # BatchSize x SampleNum
        dist_1_std = tf.tile(_cur_std[:, 0:1], multiples=[1, target_sample_num])
        dist_2_std = tf.tile(_cur_std[:, 1:2], multiples=[1, target_sample_num])

        pi = tf.constant(np.pi)
        # BatchSize x SampleNum
        prob_1 = 1 / (tf.sqrt(2 * pi) * dist_1_std) * tf.exp(
            -(target_samples - dist_1_mean) ** 2 / (2 * dist_1_std ** 2))
        prob_2 = 1 / (tf.sqrt(2 * pi) * dist_2_std) * tf.exp(
            -(target_samples - dist_2_mean) ** 2 / (2 * dist_2_std ** 2))

        prob_1 = tf.clip_by_value(prob_1, 5e-3, 3)
        prob_2 = tf.clip_by_value(prob_2, 5e-3, 3)
        prob_weight_1 = prob_1 / (prob_1 + prob_2)
        prob_weight_2 = prob_2 / (prob_1 + prob_2)

        #prob_weight_1 = tf.clip_by_value(prob_weight_1, 0.00005, 0.99995)
        #prob_weight_2 = tf.clip_by_value(prob_weight_2, 0.00005, 0.99995)
        random_matrix_1 = tf.compat.v1.random.uniform(shape=[batch_size, target_sample_num], minval=0, maxval=1)
        random_matrix_2 = tf.compat.v1.random.uniform(shape=[batch_size, target_sample_num], minval=0, maxval=1)
        #'''
        #prob_weight_1 = prob_weight_1 / tf.reduce_mean(prob_weight_1, axis=1, keepdims=True) * 0.5
        #prob_weight_2 = prob_weight_2 / tf.reduce_mean(prob_weight_2, axis=1, keepdims=True) * 0.5
        # BatchSize x SampleNum
        group_1_mask_matrix = tf.cast(random_matrix_1 < prob_weight_1, tf.float32)
        group_2_mask_matrix = tf.cast(random_matrix_2 < prob_weight_2, tf.float32)
        #'''
        # BatchSize
        group_1_num = tf.clip_by_value(tf.reduce_sum(group_1_mask_matrix, axis=1), 1, self.sample_num)
        group_2_num = tf.clip_by_value(tf.reduce_sum(group_2_mask_matrix, axis=1), 1, self.sample_num)

        # BatchSize
        sample_1_mean = tf.reduce_sum(group_1_mask_matrix * target_samples, axis=1) / \
                        group_1_num
        sample_1_std = tf.reduce_sum((target_samples - sample_1_mean[:, None]) ** 2 * group_1_mask_matrix, axis=1) / \
                       group_1_num
        sample_1_std = tf.sqrt(sample_1_std + 1e-8)
        # BatchSize
        sample_2_mean = tf.reduce_sum(group_2_mask_matrix * target_samples, axis=1) / \
                        group_2_num
        sample_2_std = tf.reduce_sum((target_samples - sample_2_mean[:, None]) ** 2 * group_2_mask_matrix, axis=1) / \
                       group_2_num
        sample_2_std = tf.sqrt(sample_2_std + 1e-8)
        # BatchSize x 2
        sample_mean = tf.concat([sample_1_mean[:, None], sample_2_mean[:, None]], axis=1)
        sample_std = tf.concat([sample_1_std[:, None], sample_2_std[:, None]], axis=1)
        sample_weight = tf.concat([group_1_num[:, None], group_2_num[:, None]], axis=1) / self.sample_num
        sample_weight = sample_weight ** 2 * 4

        return tf.stop_gradient(sample_mean), tf.stop_gradient(sample_std), tf.stop_gradient(sample_weight)

    def _build_train_op(self):
        scope = tf.compat.v1.get_default_graph().get_name_scope()
        params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                             scope=os.path.join(scope, 'Online'))
        mean_net_params, std_net_params = [], []
        for p in params:
            if 'STDPatch' in p.name:
                std_net_params.append(p)
            else:
                mean_net_params.append(p)

        batch_size = self._replay.batch_size

        # 实际执行动作的筛选索引
        # BatchSize x 1
        indices = tf.range(batch_size)[:, None]
        # BatchSize x 2
        gather_indices = tf.concat([indices, self._replay.actions[:, None]], axis=1)

        # BatchSize x 2
        _cur_mean = tf.gather_nd(self._replay_net_outputs.action_mean, gather_indices)
        # BatchSize x 2
        _cur_std = tf.gather_nd(self._replay_net_outputs.action_std, gather_indices)

        # BatchSize x 2
        _tar_mean, _tar_std, _tar_weight = self._build_target_statistic_op(_cur_mean, _cur_std)

        # BatchSize
        mean_loss = tf.reduce_mean((_cur_mean - _tar_mean) ** 2 * _tar_weight, axis=1)
        # BatchSize
        std_loss = tf.reduce_mean((_cur_std - _tar_std) ** 2 * _tar_weight, axis=1)
        # BatchSize
        w2_loss = mean_loss + std_loss
        # BatchSize x 2
        kl_loss = tf.math.log(_cur_std / _tar_std) - 0.5 + \
                  (_tar_std ** 2 + (_tar_mean - _cur_mean) ** 2) / (2 * _cur_std ** 2)
        # BatchSize
        kl_loss = tf.reduce_mean(kl_loss * _tar_weight, axis=1)

        # +++++++++++++++++++ 优先经验回放权重更新 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self._replay_scheme == 'prioritized':
            print('数据缓存器使用优先经验回放')
            prioritized_weight = mean_loss + std_loss
            update_priorities_op = self._replay.tf_set_priority(self._replay.indices,
                                                                tf.sqrt(prioritized_weight + 1e-4))
        else:
            print('数据缓存器使用均匀回放')
            update_priorities_op = tf.no_op()
        # +++++++++++++++++++ 优先经验回放权重更新 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with tf.control_dependencies([update_priorities_op]):
            if self.loss_mode == 'mse':
                mean_optimizer = tf.compat.v1.train.AdamOptimizer(0.0000625, epsilon=0.00015)
                mean_train_op = mean_optimizer.minimize(tf.reduce_mean(w2_loss))

                std_train_op = tf.no_op()
            elif self.loss_mode == 'kle':
                print('损失函数使用KL散度误差')
                mean_optimizer = tf.compat.v1.train.AdamOptimizer(0.0000625, epsilon=0.00015)
                mean_train_op = mean_optimizer.minimize(tf.reduce_mean(kl_loss))

                std_train_op = tf.no_op()
            else:
                print('损失函数使用KL散度误差 + 二阶Wasserstein误差')
                mean_optimizer = tf.compat.v1.train.AdamOptimizer(0.0000625, epsilon=0.00015)
                mean_train_op = mean_optimizer.minimize(tf.reduce_mean(w2_loss + kl_loss))

                #std_optimizer = tf.compat.v1.train.AdamOptimizer(0.000125, epsilon=0.00015)
                #std_train_op = std_optimizer.minimize(tf.reduce_mean(kl_loss))

                std_train_op = tf.no_op()

        return [mean_train_op, std_train_op,
                tf.reduce_mean(mean_loss), tf.reduce_mean(std_loss), tf.reduce_mean(kl_loss),
                _cur_mean[0], _cur_std[0], _tar_mean[0], _tar_std[0], _tar_weight[0]]

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
                _, _, mean_loss, std_loss, kl_loss, cur_mean, cur_std, tar_mean, tar_std, tar_weight = self._sess.run(self._train_op)
                if self.training_steps % self.print_freq == 0:
                    print()
                    print(f'[{self.training_steps // 1e5}]MeanLoss: {mean_loss: .2f} STDLoss: {std_loss: .2f} KLLoss: {kl_loss: .2f}')
                    print(f'[{self.training_steps // 1e5}]Mean    : {cur_mean[0]: .2f} -- {tar_mean[0]: .2f}     {cur_mean[1]: .2f} -- {tar_mean[1]: .2f}')
                    print(f'[{self.training_steps // 1e5}]STD     : {cur_std[0]: .2f} -- {tar_std[0]: .2f}     {cur_std[1]: .2f} -- {tar_std[1]: .2f}')
                    print(f'[{self.training_steps // 1e5}]WGH     : {tar_weight[0]: .2f}     {tar_weight[1]: .2f}')

                    info = self._sess.run(self.info, feed_dict={self.state_ph: self.state})
                    cur_action_mean, cur_action_std, cur_action_q, cur_action_min, cur_action_max, cur_appro_neg_q,\
                        cur_appro_pos_q = info
                    print('ActionMean-1: ', ', '.join(['%5.2f' % i for i in cur_action_mean[:, 0]]))
                    print('ActionMean-2: ', ', '.join(['%5.2f' % i for i in cur_action_mean[:, 1]]))
                    print('ActionSTD -1: ', ', '.join(['%5.2f' % i for i in cur_action_std[:, 0]]))
                    print('ActionSTD -2: ', ', '.join(['%5.2f' % i for i in cur_action_std[:, 1]]))
                    q_argmax = np.argmax(cur_action_q)
                    print('ActionQ     : ', ', '.join(['%5.2f' % i if k != q_argmax else '(%5.2f)' % i
                                                       for k, i in enumerate(cur_action_q)]))
                    '''
                    min_argmax = np.argmax(cur_action_min)
                    print('ActionMin   : ', ', '.join(['%5.2f' % i if k != min_argmax else '(%5.2f)' % i
                                                       for k, i in enumerate(cur_action_min)]))
                    max_argmax = np.argmax(cur_action_max)
                    print('ActionMax   : ', ', '.join(['%5.2f' % i if k != max_argmax else '(%5.2f)' % i
                                                       for k, i in enumerate(cur_action_max)]))
                    neg_argmax = np.argmax(cur_appro_neg_q)
                    print('ActionNeg   : ', ', '.join(['%5.2f' % i if k != neg_argmax else '(%5.2f)' % i
                                                       for k, i in enumerate(cur_appro_neg_q)]))
                    pos_argmax = np.argmax(cur_appro_pos_q)
                    print('ActionPos   : ', ', '.join(['%5.2f' % i if k != pos_argmax else '(%5.2f)' % i
                                                       for k, i in enumerate(cur_appro_pos_q)]))
                    '''
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
