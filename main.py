import numpy as np
import os
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
from dopamine.agents.implicit_quantile import implicit_quantile_agent
import tensorflow as tf


def create_iqn_agent(sess, environment, summary_writer=None):
    agent = implicit_quantile_agent.ImplicitQuantileAgent(sess, num_actions=environment.action_space.n)
    return agent


if __name__ == '__main__':
    print(f'TF_Version: {tf.__version__}  TF_Is_GPU: {tf.test.is_gpu_available()}')
    BASE_PATH = '/600lydata/zhangweijian/data/tmp/test'
    GAME = 'Asterix'
    LOG_PATH = os.path.join(BASE_PATH, 'iqn', GAME)

    random_dqn_config = """
    import dopamine.discrete_domains.atari_lib
    import dopamine.discrete_domains.run_experiment
    atari_lib.create_atari_environment.game_name = '{}'
    atari_lib.create_atari_environment.sticky_actions = True
    run_experiment.Runner.num_iterations = 200
    run_experiment.Runner.training_steps = 10
    run_experiment.Runner.max_steps_per_episode = 100
    """.format(GAME)

    gin.parse_config(random_dqn_config, skip_unknown=False)
    random_dqn_runner = run_experiment.TrainRunner(LOG_PATH, create_iqn_agent)
    print('Will train agent, please be patient, may be a while...')
    random_dqn_runner.run_experiment()
    print('Done training!')

