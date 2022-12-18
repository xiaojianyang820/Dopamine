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
class Double_Weight_Gaussian_V2_Network(keras.Model):
    Double_Gaussian_NetworkType = collections.namedtuple(
        'Double_Gaussian_Network',
        ['action_mean', 'action_std', 'action_weight']
    )

    def __init__(self, action_num: int, name: str = None, embedding_dim: int = 128):
        super(Double_Weight_Gaussian_V2_Network, self).__init__(name=name)
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
                                                 activation=None, name='Mean')

        self.dense_2_std_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                                 activation=activation_fn, name='STDFeature')
        self.dense_2_std_2 = keras.layers.Dense(action_num * 1, kernel_initializer=kernel_initializer,
                                                 activation=None, name='STD')
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
                                                 activation=None, name='S_Mean')

        self.s_dense_2_std_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                                activation=activation_fn, name='S_STDFeature')
        self.s_dense_2_std_2 = keras.layers.Dense(action_num * 1, kernel_initializer=kernel_initializer,
                                                activation=None, name='S_STD')
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.dense_3_1 = keras.layers.Dense(action_num * 5, kernel_initializer=kernel_initializer,
                                            activation=activation_fn, name='WeightFeature')
        self.dense_3_2 = keras.layers.Dense(action_num, kernel_initializer=kernel_initializer,
                                            activation=None, name='Weight')

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
            #action_extra_mean = tf.math.softplus(action_extra_mean)
            # BatchSize x ActionNum x 1
            #action_hv_mean = tf.reshape(action_extra_mean + action_mean_cache, [batch_size, self.action_num])[:, :, None]
            action_hv_mean = tf.reshape(action_extra_mean, [batch_size, self.action_num])[:, :, None]

        # 特征向量 -->> 对数标准差对 区块
        with tf.compat.v1.name_scope('STDPatch'):
            # BatchSize x [ActionNum * 1]
            action_std = self.dense_2_std_2(self.dense_2_std_1(state_feature_vec))
            # BatchSize x ActionNum x 1
            action_std = tf.reshape(action_std, [batch_size, self.action_num])[:, :, None] / 2
            action_std = tf.exp(action_std)

            # BatchSize x [ActionNum * 1]
            action_hv_std = self.s_dense_2_std_2(self.s_dense_2_std_1(state_hv_feature_vec))
            # BatchSize x ActionNum x 1
            action_hv_std = tf.reshape(action_hv_std, [batch_size, self.action_num])[:, :, None] / 2
            action_hv_std = tf.exp(action_hv_std)

        # 特征向量 -->> 分布权重 区块
        with tf.compat.v1.name_scope('WeightPatch'):
            # BatchSize x ActionNum
            action_weight = self.dense_3_2(self.dense_3_1(state_feature_vec + state_hv_feature_vec))
            action_weight = tf.math.sigmoid(action_weight / 10)

        # BatchSize x ActionNum x 2
        action_mean = tf.concat([action_mean, action_hv_mean], axis=2)
        action_std = tf.concat([action_std, action_hv_std], axis=2)

        return self.Double_Gaussian_NetworkType(
            action_mean=action_mean, action_std=action_std, action_weight=action_weight
        )


@gin.configurable
class DoubleWeightGaussianAgent(rainbow_agent.RainbowAgent):
    def __init__(self, sess, action_num, network=Double_Weight_Gaussian_V2_Network, double_dqn: bool = True,
                 sample_num: int = 128, action_mode: str = 'mean', loss_mode: str = 'mse',
                 summary_writer=None, summary_writing_frequency=500, print_freq=100000):
        self.action_num = action_num
        self.double_dqn = double_dqn
        self.sample_num = sample_num
        self.print_freq = print_freq
        self.action_mode = action_mode
        self.loss_mode = loss_mode

        super(DoubleWeightGaussianAgent, self).__init__(
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
        # 当前状态下动作回报的标准差对
        # BatchSize x ActionNum x 2
        self.cur_action_std = self._action_node_outputs.action_std
        # 当前状态下动作回报的权重分配
        # BatchSize x ActionNum
        self.cur_action_weight = self._action_node_outputs.action_weight
        # BatchSize x ActionNum x SampleNum
        self.cur_sample = self._build_samples_op(self.cur_action_mean, self.cur_action_std, self.cur_action_weight)
        # BatchSize x ActionNum
        self.cur_sample_mean = tf.reduce_mean(self.cur_sample, axis=2)
        # BatchSize x ActionNum
        self.cur_sample_dev = tf.reduce_mean((self.cur_sample - self.cur_sample_mean[:, :, None]) ** 2, axis=2)
        self.cur_sample_std = tf.sqrt(self.cur_sample_dev)
        # 当前状态下动作回报的数学期望
        # BatchSize x ActionNum x 2
        cur_action_weight = tf.concat([self.cur_action_weight[:, :, None], 1 - self.cur_action_weight[:, :, None]],
                                      axis=2)
        # BatchSize x ActionNum
        self.cur_action_q = tf.reduce_sum(self.cur_action_mean * cur_action_weight, axis=2)
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
            online_action_weight = _replay_online_network_outputs.action_weight
            online_action_weight = tf.concat([online_action_weight[:, :, None], 1 - online_action_weight[:, :, None]],
                                             axis=2)
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
            target_action_weight = self._replay_target_network_outputs.action_weight
            target_action_weight = tf.concat([target_action_weight[:, :, None], 1 - target_action_weight[:, :, None]],
                                             axis=2)
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

    def _build_samples_op(self, _target_mean, _target_std, _target_weight):
        batch_size = self._replay.batch_size
        sample_num = self.sample_num
        if len(_target_mean.shape) == 2:
            # BatchSize x SampleNum
            dist_mask_1 = tf.range(start=0, limit=sample_num, dtype=tf.int32)[None, :] <\
                              tf.cast(sample_num * _target_weight, dtype=tf.int32)[:, None]
            dist_mask_1 = tf.cast(dist_mask_1, tf.float32)
            dist_mask_2 = 1 - dist_mask_1

            # BatchSize x SampleNum
            mean_1 = tf.tile(_target_mean[:, 0:1], multiples=[1, sample_num])
            # BatchSize x SampleNum
            mean_2 = tf.tile(_target_mean[:, 1:2], multiples=[1, sample_num])
            # BatchSize x SampleNum
            std_1 = tf.tile(_target_std[:, 0:1], multiples=[1, sample_num])
            # BatchSize x SampleNum
            std_2 = tf.tile(_target_std[:, 1:2], multiples=[1, sample_num])
            # BatchSize x SampleNum
            sample_1 = tf.compat.v1.random.normal(shape=[batch_size, sample_num], mean=mean_1, stddev=std_1)
            sample_2 = tf.compat.v1.random.normal(shape=[batch_size, sample_num], mean=mean_2, stddev=std_2)
        else:
            # BatchSize x ActionNum x SampleNum
            dist_mask_1 = tf.range(start=0, limit=sample_num, dtype=tf.int32)[None, None, :] < \
                          tf.cast(sample_num * _target_weight, dtype=tf.int32)[:, :, None]
            dist_mask_1 = tf.cast(dist_mask_1, tf.float32)
            dist_mask_2 = 1 - dist_mask_1
            # BatchSize x ActionNum x [SampleNum // 2]
            mean_1 = tf.tile(_target_mean[:, :, 0:1], multiples=[1, 1, sample_num])
            # BatchSize x ActionNum x [SampleNum // 2]
            mean_2 = tf.tile(_target_mean[:, :, 1:2], multiples=[1, 1, sample_num])
            # BatchSize x ActionNum x [SampleNum // 2]
            std_1 = tf.tile(_target_std[:, :, 0:1], multiples=[1, 1, sample_num])
            # BatchSize x ActionNum x [SampleNum // 2]
            std_2 = tf.tile(_target_std[:, :, 1:2], multiples=[1, 1, sample_num])
            # BatchSize x ActionNum x [SampleNum // 2]
            sample_1 = tf.compat.v1.random.normal(shape=[batch_size, self.action_num, sample_num], mean=mean_1, stddev=std_1)
            sample_2 = tf.compat.v1.random.normal(shape=[batch_size, self.action_num, sample_num], mean=mean_2, stddev=std_2)

        # BatchSize x ... x SampleNum
        sample = sample_1 * dist_mask_1 + sample_2 * dist_mask_2
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

        # BatchSize x 2
        _target_mean = tf.gather_nd(self._replay_target_network_outputs.action_mean, gather_indices)
        # BatchSize x 2
        _target_std = tf.gather_nd(self._replay_target_network_outputs.action_std, gather_indices)
        # BatchSize
        _target_weight = tf.gather_nd(self._replay_target_network_outputs.action_weight, gather_indices)

        sample = self._build_samples_op(_target_mean, _target_std, _target_weight)

        return rewards[:, None] + gamma_with_terminal[:, None] * sample

    def _build_target_statistic_op(self, _cur_mean, _cur_std, _cur_weight):
        # BatchSize x SampleNum
        target_samples = self._build_target_samples_op()
        # E步
        # 每一个样本归属到每一个类别的概率
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

        prob_1 = tf.clip_by_value(prob_1, 1e-4, 2)
        prob_2 = tf.clip_by_value(prob_2, 1e-4, 2)

        total_weight = prob_1 * _cur_weight[:, 0:1] + prob_2 * _cur_weight[:, 1:2]
        # BatchSize x SampleNum
        weight_1 = prob_1 * _cur_weight[:, 0:1] / total_weight
        weight_2 = prob_2 * _cur_weight[:, 1:2] / total_weight

        weight_1 = tf.stop_gradient(weight_1)
        weight_2 = tf.stop_gradient(weight_2)
        # M步
        # BatchSize
        weight_1_sum = tf.reduce_sum(weight_1, axis=1)
        weight_2_sum = tf.reduce_sum(weight_2, axis=1)
        # BatchSize
        _tar_mean_1 = tf.reduce_sum(weight_1 * target_samples, axis=1) / weight_1_sum
        _tar_mean_2 = tf.reduce_sum(weight_2 * target_samples, axis=1) / weight_2_sum
        # BatchSize x 2
        _tar_mean = tf.concat([_tar_mean_1[:, None], _tar_mean_2[:, None]], axis=1)

        avg_1_mean = _tar_mean_1[:, None] * 0.8 + dist_1_mean * 0.2
        avg_2_mean = _tar_mean_2[:, None] * 0.8 + dist_2_mean * 0.2
        _tar_sigma_1 = tf.reduce_sum(weight_1 * (target_samples - avg_1_mean) ** 2, axis=1) / weight_1_sum
        _tar_sigma_2 = tf.reduce_sum(weight_2 * (target_samples - avg_2_mean) ** 2, axis=1) / weight_1_sum
        # BatchSize x 2
        _tar_sigma = tf.concat([_tar_sigma_1[:, None], _tar_sigma_2[:, None]], axis=1)
        _tar_std = tf.sqrt(_tar_sigma)

        _tar_weight_1 = weight_1_sum / self.sample_num
        _tar_weight_2 = weight_2_sum / self.sample_num
        # BatchSize x 2
        _tar_weight = tf.concat([_tar_weight_1[:, None], _tar_weight_2[:, None]], axis=1)

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

        # BatchSize x 2
        _cur_mean = tf.gather_nd(self._replay_net_outputs.action_mean, gather_indices)
        # BatchSize x 2
        _cur_std = tf.gather_nd(self._replay_net_outputs.action_std, gather_indices)
        # BatchSize
        _cur_weight = tf.gather_nd(self._replay_net_outputs.action_weight, gather_indices)
        # BatchSize x 2
        _cur_weight = tf.concat([_cur_weight[:, None], 1 - _cur_weight[:, None]], axis=1)

        # BatchSize x 2
        _tar_mean, _tar_std, _tar_weight = self._build_target_statistic_op(_cur_mean, _cur_std, _cur_weight)

        # BatchSize
        mean_loss = tf.reduce_sum((_cur_mean - _tar_mean) ** 2 * _tar_weight, axis=1)
        # BatchSize
        std_loss = tf.reduce_sum((_cur_std - _tar_std) ** 2 * _tar_weight, axis=1)
        # BatchSize
        weight_loss = tf.reduce_sum(tf.square(_cur_weight - _tar_weight), axis=1)

        # +++++++++++++++++++ 优先经验回放权重更新 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self._replay_scheme == 'prioritized':
            print('数据缓存器使用优先经验回放')
            prioritized_weight = mean_loss + std_loss + weight_loss
            update_priorities_op = self._replay.tf_set_priority(self._replay.indices,
                                                                tf.sqrt(prioritized_weight + 1e-4))
        else:
            print('数据缓存器使用均匀回放')
            update_priorities_op = tf.no_op()
        # +++++++++++++++++++ 优先经验回放权重更新 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with tf.control_dependencies([update_priorities_op]):
            if self.loss_mode == 'mse':
                w2_optimizer = tf.compat.v1.train.AdamOptimizer(0.0000625, epsilon=0.00015)
                w2_train_op = w2_optimizer.minimize(tf.reduce_mean(mean_loss + std_loss))

                weight_optimizer = tf.compat.v1.train.AdagradOptimizer(0.0000625)
                weight_train_op = weight_optimizer.minimize(tf.reduce_mean(weight_loss))


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
                    print(f'[{self.training_steps // 1e5}]Mean: {cur_mean[0]: .2f} -- {tar_mean[0]: .2f}     {cur_mean[1]: .2f} -- {tar_mean[1]: .2f}')
                    print(f'[{self.training_steps // 1e5}]STD: {cur_std[0]: .2f} -- {tar_std[0]: .2f}     {cur_std[1]: .2f} -- {tar_std[1]: .2f}')
                    print(f'[{self.training_steps // 1e5}]Weight: {cur_weight[0]: .2f} -- {tar_weight[0]: .2f}')

                    info = self._sess.run(self.info, feed_dict={self.state_ph: self.state})
                    cur_action_mean, cur_action_std, cur_action_q, cur_action_min, cur_action_max, cur_action_weight = info
                    print('ActionMean-1: ', ', '.join(['%5.2f' % i for i in cur_action_mean[:, 0]]))
                    print('ActionMean-2: ', ', '.join(['%5.2f' % i for i in cur_action_mean[:, 1]]))
                    print('ActionSTD -1: ', ', '.join(['%5.2f' % i for i in cur_action_std[:, 0]]))
                    print('ActionSTD -2: ', ', '.join(['%5.2f' % i for i in cur_action_std[:, 1]]))
                    print('ActionWgt -2: ', ', '.join(['%5.2f' % i for i in cur_action_weight]))
                    q_argmax = np.argmax(cur_action_q)
                    print('ActionQ     : ', ', '.join(['%5.2f' % i if k != q_argmax else '(%5.2f)' % i
                                                       for k, i in enumerate(cur_action_q)]))
                    min_argmax = np.argmax(cur_action_min)
                    print('ActionMin   : ', ', '.join(['%5.2f' % i if k != min_argmax else '(%5.2f)' % i
                                                       for k, i in enumerate(cur_action_min)]))
                    max_argmax = np.argmax(cur_action_max)
                    print('ActionMax   : ', ', '.join(['%5.2f' % i if k != max_argmax else '(%5.2f)' % i
                                                       for k, i in enumerate(cur_action_max)]))

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

