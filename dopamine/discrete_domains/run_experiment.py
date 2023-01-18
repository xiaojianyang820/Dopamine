# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.agents.mg.multi_weight_gaussian_agent import MultiWeightGaussianAgent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger
from dopamine.jax.agents.dqn import dqn_agent as jax_dqn_agent
from dopamine.jax.agents.full_rainbow import full_rainbow_agent
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent as jax_implicit_quantile_agent
from dopamine.jax.agents.quantile import quantile_agent as jax_quantile_agent
from dopamine.jax.agents.rainbow import rainbow_agent as jax_rainbow_agent
from dopamine.metrics import collector_dispatcher
from dopamine.metrics import statistics_instance
import gin.tf
import numpy as np
import tensorflow as tf


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.compat.v1.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'mwg':
      return MultiWeightGaussianAgent(sess, action_num=environment.action_space.n, summary_writer=summary_writer)
  elif agent_name == 'jax_dqn':
    return jax_dqn_agent.JaxDQNAgent(num_actions=environment.action_space.n,
                                     summary_writer=summary_writer)
  elif agent_name == 'jax_quantile':
    return jax_quantile_agent.JaxQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_rainbow':
    return jax_rainbow_agent.JaxRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'full_rainbow':
    return full_rainbow_agent.JaxFullRainbowAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'jax_implicit_quantile':
    return jax_implicit_quantile_agent.JaxImplicitQuantileAgent(
        num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    schedule: string, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class Runner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  处理执行Dopamine试验的对象

  这里我们使用术语'experiment'来指代智能体与环境之间的模拟交互，以及这些交互过程中所产生的统计数据

  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               clip_rewards=True,
               use_legacy_logger=True,
               fine_grained_print_to_console=True):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      clip_rewards: bool, whether to clip rewards in [-1, 1].
      use_legacy_logger: bool, whether to use the legacy Logger. This will be
        deprecated soon, replaced with the new CollectorDispatcher setup.
      fine_grained_print_to_console: bool, whether to print fine-grained
        progress to console (useful for debugging).

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.compat.v1.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.

    初始化一个管理一个完整试验的Runner对象
    Args:
        base_dir: str,
            管理全部目标子文件夹的基础文件夹
        create_agent_fn: callable,
            一个将一个Tensorflow会话和一个环境作为输入参数，返回一个智能体的函数
        create_environment_fn: callable,
            一个函数，输入为一个问题名称，然后返回相应的Gym环境
        checkpoint_file_prefix: str,
            存储节点文件的名称前缀
        logging_file_prefix: str,
            日志文件的名称前缀
        log_every_n: int,
            日志文件记录的频率
        num_iterations: int,
            总共的迭代次数
        training_steps: int,
            训练的步数
        evaluation_steps: int,
            评估的步数
        max_steps_per_episode:int,
            在一个回合中最大的会话次数
        clip_rewards: bool,
            是否将奖励截断到[-1, 1]之间
        use_legacy_logger: bool,
            是否使用遗产日志记录器。这一功能很快就会被废弃，替换成一种新的CoolectorDispatcher设置
        fine_grained_print_to_console: bool,
            是否将细粒度的过程日志打印到控制台（方便进行调试）
    初始化构造器会按照如下步骤进行操作：
        - 初始化一个环境
        - 初始化一个计算资源会话
        - 初始化一个日志记录器
        - 初始化一个智能体
        - 如果可能的话，从最近一次的存储节点恢复数据。初始化存储节点对象。
    """
    assert base_dir is not None
    # 是否使用遗产日志记录器
    self._legacy_logger_enabled = use_legacy_logger
    # 是否将细粒度的处理过程日志打印到控制台
    self._fine_grained_print_to_console_enabled = fine_grained_print_to_console
    # 日志文件的名称前缀
    self._logging_file_prefix = logging_file_prefix
    # 日志文件的记录频率
    self._log_every_n = log_every_n
    # 迭代总数
    self._num_iterations = num_iterations
    # 训练步数
    self._training_steps = training_steps
    # 评估步数
    self._evaluation_steps = evaluation_steps
    # 每一个回合中的最大会话次数
    self._max_steps_per_episode = max_steps_per_episode
    # 存储日志基文件夹
    self._base_dir = base_dir
    # 是否截断奖励信号到[-1, 1]之间
    self._clip_rewards = clip_rewards
    # 创建存储节点文件夹和日志记录器
    self._create_directories()
    # 创建环境对象
    self._environment = create_environment_fn()
    # 目前的版本是由智能体来管理计算资源会话
    self._sess = None
    # We're using a bit of a hack in that we pass in _base_dir instead of an
    # actually SummaryWriter. This is because the agent is now in charge of the
    # session, but needs to create the SummaryWriter before creating the ops,
    # and in order to do so, it requires the base directory.
    # 这里的技巧是向summary_writer参数传递了一个基文件夹地址，而不是一个实际的SummaryWriter，
    # 这是因为当前由智能体去管理计算资源会话，但是却需要在创建张量之前就构造出SummaryWriter，
    # 出于这一原因，所以它需要一个基文件夹
    # 构建智能体
    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=self._base_dir)
    # 如果主函数没有创建计算资源会话的话，那么就由智能体创建计算资源会话
    if hasattr(self._agent, '_sess'):
      self._sess = self._agent._sess
    # SummaryWriter也使用智能体的SummaryWriter
    self._summary_writer = self._agent.summary_writer
    # 检查存储节点文件夹中是否有已完成的存储节点文件，如果有的话，将相应信息重新加载到智能体当中
    # 迭代序号从已有的存储节点文件开始计数
    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    # Create a collector dispatcher for metrics reporting.
    # 创建一个度量指标收集分发器
    self._collector_dispatcher = collector_dispatcher.CollectorDispatcher(
        self._base_dir)
    set_collector_dispatcher_fn = getattr(
        self._agent, 'set_collector_dispatcher', None)
    if callable(set_collector_dispatcher_fn):
      set_collector_dispatcher_fn(self._collector_dispatcher)

  @property
  def _use_legacy_logger(self):
    if not hasattr(self, '_legacy_logger_enabled'):
      return True
    return self._legacy_logger_enabled

  @property
  def _fine_grained_print_to_console(self):
    if not hasattr(self, '_fine_grained_print_to_console_enabled'):
      return True
    return self._fine_grained_print_to_console_enabled

  def _create_directories(self):
    """
    Create necessary sub-directories.
    创建必要的子文件夹
    """
    # 创建存储节点文件夹
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    # 是否使用日志记录器，如果使用的话
    if self._use_legacy_logger:
      logging.warning(
          'DEPRECATION WARNING: Logger is being deprecated. '
          'Please switch to CollectorDispatcher!')
      # 创建一个日志记录器
      self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.

    如果之前就有存储节点数据的话，就载入最新的存储节点

    这个方法首先创建一个存储节点对象，然后调用相应的`checkpointer.get_latest_checkpoint_number'方法来判断
    在存储节点文件夹中是否有合法的存储节点数据，以及最新的节点编号是多少。如果有一个合法的存储节点，该方法就会加载
    相应的数据，并将这些数据转发给Agent。如果智能体成功解析了这些数据，那么该方法就会验证这些数据中是否包括键“logs”和
    “current_iteration”。该方法会从这些数据里面加载”Logger“数据，然后返回当前的迭代次数”current_iteration"。

    Args:
        checkpoint_file_prefix: str,
            存储节点文件的名称前缀

    Returns:
        start_iteration: int,
            试验开始的迭代编号
        experiment_checkpointer: callable,
            试验的存储节点对象
    """
    # 构造一个存储节点文件管理对象
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    # 检查存储节点文件是否存在，注意到，比如说存储节点文件0存在，那就意味着已经完成了第0次计算，需要从第1次计算重新开始
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          if self._use_legacy_logger:
            self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        logging.info('Reloaded checkpoint and will start from iteration %d',
                     self._start_iteration)

  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward, terminal=True):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
      terminal: bool, whether the last state-action led to a terminal state.
    """
    if isinstance(self._agent, jax_dqn_agent.JaxDQNAgent):
      self._agent.end_episode(reward, terminal)
    else:
      # TODO(joshgreaves): Add terminal signal to TF dopamine agents
      self._agent.end_episode(reward)

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      if self._clip_rewards:
        # Perform reward clipping.
        #print('将奖励信号裁剪到-1至+1')
        reward = np.clip(reward, -1, 1)
      else:
        #print('调整奖励信号的量纲')
        #reward = np.sign(reward) * np.sqrt(np.abs(reward)) + 0.001 * reward
        reward = reward
        pass

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._end_episode(reward, is_terminal)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    self._end_episode(reward, is_terminal)

    return step_number, total_reward

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.

    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      if self._fine_grained_print_to_console:
        # We use sys.stdout.write instead of logging so as to flush frequently
        # without generating a line break.
        sys.stdout.write('Steps executed: {} '.format(step_count) +
                         'Episode length: {} '.format(episode_length) +
                         'Return: {}\r'.format(episode_return))
        sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per second.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    average_steps_per_second = number_steps / time_delta
    statistics.append(
        {'train_average_steps_per_second': average_steps_per_second})
    logging.info('Average undiscounted return per training episode: %.2f',
                 average_return)
    logging.info('Average training steps per second: %.2f',
                 average_steps_per_second)
    return num_episodes, average_return, average_steps_per_second

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    logging.info('Average undiscounted return per evaluation episode: %.2f',
                 average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)

    self._collector_dispatcher.write([
        statistics_instance.StatisticsInstance('Train/NumEpisodes',
                                               num_episodes_train,
                                               iteration),
        statistics_instance.StatisticsInstance('Train/AverageReturns',
                                               average_reward_train,
                                               iteration),
        statistics_instance.StatisticsInstance('Train/AverageStepsPerSecond',
                                               average_steps_per_second,
                                               iteration),
        statistics_instance.StatisticsInstance('Eval/NumEpisodes',
                                               num_episodes_eval,
                                               iteration),
        statistics_instance.StatisticsInstance('Eval/AverageReturns',
                                               average_reward_eval,
                                               iteration),
    ])
    if self._summary_writer is not None:
      self._save_tensorboard_summaries(iteration, num_episodes_train,
                                       average_reward_train, num_episodes_eval,
                                       average_reward_eval,
                                       average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval,
                                  average_steps_per_second):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
      average_steps_per_second: float, The average number of steps per second.
    """
    if self._summary_writer is None:
      return

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar('Train/NumEpisodes', num_episodes_train,
                          step=iteration)
        tf.summary.scalar('Train/AverageReturns', average_reward_train,
                          step=iteration)
        tf.summary.scalar('Train/AverageStepsPerSecond',
                          average_steps_per_second, step=iteration)
        tf.summary.scalar('Eval/NumEpisodes', num_episodes_eval, step=iteration)
        tf.summary.scalar('Eval/AverageReturns', average_reward_eval,
                          step=iteration)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='Train/NumEpisodes', simple_value=num_episodes_train),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageReturns', simple_value=average_reward_train),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageStepsPerSecond',
              simple_value=average_steps_per_second),
          tf.compat.v1.Summary.Value(
              tag='Eval/NumEpisodes', simple_value=num_episodes_eval),
          tf.compat.v1.Summary.Value(
              tag='Eval/AverageReturns', simple_value=average_reward_eval)
      ])
      self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    if not hasattr(self, '_logger'):
      return

    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      if self._use_legacy_logger:
        experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      if self._use_legacy_logger:
        self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)
      self._collector_dispatcher.flush()
    if self._summary_writer is not None:
      self._summary_writer.flush()
    self._collector_dispatcher.close()


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))

    self._collector_dispatcher.write([
        statistics_instance.StatisticsInstance('Train/NumEpisodes',
                                               num_episodes_train,
                                               iteration),
        statistics_instance.StatisticsInstance('Train/AverageReturns',
                                               average_reward_train,
                                               iteration),
        statistics_instance.StatisticsInstance('Train/AverageStepsPerSecond',
                                               average_steps_per_second,
                                               iteration),
    ])
    if self._summary_writer is not None:
      self._save_tensorboard_summaries(iteration, num_episodes_train,
                                       average_reward_train,
                                       average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, average_steps_per_second):
    """Save statistics as tensorboard summaries."""
    if self._summary_writer is None:
      return

    if self._sess is None:
      with self._summary_writer.as_default():
        tf.summary.scalar('Train/NumEpisodes', num_episodes, step=iteration)
        tf.summary.scalar('Train/AverageReturns', average_reward,
                          step=iteration)
        tf.summary.scalar('Train/AverageStepsPerSecond',
                          average_steps_per_second, step=iteration)
      self._summary_writer.flush()
    else:
      summary = tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
              tag='Train/NumEpisodes', simple_value=num_episodes),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageReturns', simple_value=average_reward),
          tf.compat.v1.Summary.Value(
              tag='Train/AverageStepsPerSecond',
              simple_value=average_steps_per_second),
      ])
      self._summary_writer.add_summary(summary, iteration)
