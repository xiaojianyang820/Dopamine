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
"""A checkpointing mechanism for Dopamine agents.

This Checkpointer expects a base directory where checkpoints for different
iterations are stored. Specifically, Checkpointer.save_checkpoint() takes in
as input a dictionary 'data' to be pickled to disk. At each iteration, we
write a file called 'cpkt.#', where # is the iteration number. The
Checkpointer also cleans up old files, maintaining up to the
`checkpoint_duration` most recent iterations.

The Checkpointer writes a sentinel file to indicate that checkpointing was
globally successful. This means that all other checkpointing activities
(saving the Tensorflow graph, the replay buffer) should be performed *prior*
to calling Checkpointer.save_checkpoint(). This allows the Checkpointer to
detect incomplete checkpoints.

#### Example

After running 10 iterations (numbered 0...9) with base_directory='/checkpoint',
the following files will exist:
```
  /checkpoint/cpkt.6
  /checkpoint/cpkt.7
  /checkpoint/cpkt.8
  /checkpoint/cpkt.9
  /checkpoint/sentinel_checkpoint_complete.6
  /checkpoint/sentinel_checkpoint_complete.7
  /checkpoint/sentinel_checkpoint_complete.8
  /checkpoint/sentinel_checkpoint_complete.9
```
Dopamine智能体的存储节点机制

这个Checkpointer期望一个基文件夹地址，不同迭代编号的节点数据会存储在这个文件夹里。特殊来说，Checkpointer.save_checkpoint()读取一个
'data'文件夹作为输入，然后序列化到硬盘。在每一次迭代中，我们写一个文件叫做'ckpt.#'，这里的#是迭代次数。Checkpointer会清理旧文件，以
保证文件数量保持在有限的数量上。

Checkpointer会写入一个“哨兵”文件来表明存储节点是不是完全处理成功的。这就意味着全部的其他存储节点操作（存储TensorFlow计算图，数据缓存器）
应该先于调用Checkpointer.save_checkpoint()进行调用。这使得Checkpointer可以去检测不完全的存储节点。

#### 案例
在执行10轮迭代之后（0...9），如果基文件夹是'/checkpoint'，那么这个文件夹里面就会有如下的文件：
```
  /checkpoint/cpkt.6
  /checkpoint/cpkt.7
  /checkpoint/cpkt.8
  /checkpoint/cpkt.9
  /checkpoint/sentinel_checkpoint_complete.6
  /checkpoint/sentinel_checkpoint_complete.7
  /checkpoint/sentinel_checkpoint_complete.8
  /checkpoint/sentinel_checkpoint_complete.9
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import logging

import gin
import tensorflow as tf


@gin.configurable
def get_latest_checkpoint_number(base_directory,
                                 override_number=None,
                                 sentinel_file_identifier='checkpoint'):
  """Returns the version number of the latest completed checkpoint.

  Args:
    base_directory: str, directory in which to look for checkpoint files.
    override_number: None or int, allows the user to manually override
      the checkpoint number via a gin-binding.
    sentinel_file_identifier: str, prefix used by checkpointer for naming
      sentinel files.

  Returns:
    int, the iteration number of the latest checkpoint, or -1 if none was found.
  """
  if override_number is not None:
    return override_number

  sentinel = 'sentinel_{}_complete.*'.format(sentinel_file_identifier)
  glob = os.path.join(base_directory, sentinel)
  def extract_iteration(x):
    return int(x[x.rfind('.') + 1:])
  try:
    checkpoint_files = tf.io.gfile.glob(glob)
  except tf.errors.NotFoundError:
    return -1
  try:
    latest_iteration = max(extract_iteration(x) for x in checkpoint_files)
    return latest_iteration
  except ValueError:
    return -1


@gin.configurable
class Checkpointer(object):
  """
  Class for managing checkpoints for Dopamine agents.
  管理Dopamine智能体存储节点的类
  """

  def __init__(self, base_directory, checkpoint_file_prefix='ckpt',
               sentinel_file_identifier='checkpoint', checkpoint_frequency=1,
               checkpoint_duration=4,
               keep_every=None):
    """Initializes Checkpointer.

    Args:
      base_directory: str, directory where all checkpoints are saved/loaded.
      checkpoint_file_prefix: str, prefix to use for naming checkpoint files.
      sentinel_file_identifier: str, prefix to use for naming sentinel files.
      checkpoint_frequency: int, the frequency at which to checkpoint.
      checkpoint_duration: int, how many checkpoints to keep
      keep_every: Optional (int or None), keep all checkpoints == 0 % this
        number. Set to None to disable.

    Raises:
      ValueError: if base_directory is empty, or not creatable.

    初始化一个Checkpointer

    Args:
      base_directory: str,
        存储或者读取存储节点文件的文件夹
      checkpoint_file_prefix: str,
        存储节点文件的名称前缀
      sentinel_file_identifier: str,
        哨兵文件的名称前缀
      checkpoint_frequency: int,
        存储文件的频率
      checkpoint_duration: int,
        存储多少个存储节点文件
      keep_every: int, optional=None,
        长期保存某一个频率的存储节点文件，如果设置为None，就不使用这个功能
    """
    # 如果没有基文件夹，那么就报错
    if not base_directory:
      raise ValueError('No path provided to Checkpointer.')
    # 存储节点文件的前缀
    self._checkpoint_file_prefix = checkpoint_file_prefix
    # 哨兵文件的前缀
    self._sentinel_file_prefix = 'sentinel_{}_complete'.format(
        sentinel_file_identifier)
    # 存储节点文件的存储频率
    self._checkpoint_frequency = checkpoint_frequency
    # 存储节点文件的存储数量
    self._checkpoint_duration = checkpoint_duration
    # 是否固定存储某一个频率的存储节点文件
    self._keep_every = keep_every
    # 基文件夹
    self._base_directory = base_directory
    # 创建基文件夹
    try:
      tf.io.gfile.makedirs(base_directory)
    except tf.errors.PermissionDeniedError as permission_error:
      # We catch the PermissionDeniedError and issue a more useful exception.
      raise ValueError('Unable to create checkpoint path: {}.'.format(
          base_directory)) from permission_error

  def _generate_filename(self, file_prefix, iteration_number) -> str:
    """
    Returns a checkpoint filename from prefix and iteration number.
    基于名称前缀和迭代序号返回存储节点文件名称
    Args:
      file_prefix: str,
        存储节点文件前缀
      iteration_number: int,
        迭代序号
    Return: str,
      存储节点文件名称
    """
    filename = '{}.{}'.format(file_prefix, iteration_number)
    return os.path.join(self._base_directory, filename)

  def _save_data_to_file(self, data, filename):
    """
    Saves the given 'data' object to a file.
    将给定的数据存储到文件中
    """
    with tf.io.gfile.GFile(filename, 'w') as fout:
      pickle.dump(data, fout)

  def save_checkpoint(self, iteration_number, data):
    """Saves a new checkpoint at the current iteration_number.

    Args:
      iteration_number: int, the current iteration number for this checkpoint.
      data: Any (picklable) python object containing the data to store in the
        checkpoint.
    """
    # 如果迭代序号不是目标频率的倍数，就不进行存储
    if iteration_number % self._checkpoint_frequency != 0:
      return
    # 生成存储节点文件的名称
    filename = self._generate_filename(self._checkpoint_file_prefix,
                                       iteration_number)
    # 将存储节点文件写入到目标文件中
    self._save_data_to_file(data, filename)
    # 生成哨兵文件的名称
    filename = self._generate_filename(self._sentinel_file_prefix,
                                       iteration_number)
    # 在哨兵文件中写入成功标记
    with tf.io.gfile.GFile(filename, 'wb') as fout:
      fout.write('done')
    # 清理之前的存储节点文件
    self._clean_up_old_checkpoints(iteration_number)

  def _clean_up_old_checkpoints(self, iteration_number):
    """Removes sufficiently old checkpoints."""
    # After writing a the checkpoint and sentinel file, we garbage collect files
    # that are self._checkpoint_duration * self._checkpoint_frequency
    # versions old.
    stale_iteration_number = iteration_number - (self._checkpoint_frequency *
                                                 self._checkpoint_duration)

    # If keep_every has been set, we spare every keep_every'th checkpoint
    if (self._keep_every is not None
        and (stale_iteration_number %
             (self._keep_every*self._checkpoint_frequency) == 0)):
      return

    if stale_iteration_number >= 0:
      stale_file = self._generate_filename(self._checkpoint_file_prefix,
                                           stale_iteration_number)
      stale_sentinel = self._generate_filename(self._sentinel_file_prefix,
                                               stale_iteration_number)
      try:
        tf.io.gfile.remove(stale_file)
        tf.io.gfile.remove(stale_sentinel)
      except tf.errors.NotFoundError:
        # Ignore if file not found.
        logging.info('Unable to remove %s or %s.', stale_file, stale_sentinel)

  def _load_data_from_file(self, filename):
    if not tf.io.gfile.exists(filename):
      return None
    with tf.io.gfile.GFile(filename, 'rb') as fin:
      return pickle.load(fin)

  def load_checkpoint(self, iteration_number):
    """Tries to reload a checkpoint at the selected iteration number.

    Args:
      iteration_number: The checkpoint iteration number to try to load.

    Returns:
      If the checkpoint files exist, two unpickled objects that were passed in
        as data to save_checkpoint; returns None if the files do not exist.
    """
    checkpoint_file = self._generate_filename(self._checkpoint_file_prefix,
                                              iteration_number)
    return self._load_data_from_file(checkpoint_file)
