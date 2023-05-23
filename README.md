# Dopamine
<div align="center">
  <img src="https://google.github.io/dopamine/images/dopamine_logo.png"><br><br>
</div>

## English Document
This project will further develop the Distributional Reinforcement Learning(DRL) theory based on the Google open source project [dopamine](https://github.com/google/dopamine). For related dependencies and additional information of this project, please refer to the dopamine project documentation

### New Algorithm
### Mixed Weighted Gaussian Model(MWG)
Under the Dopamine framework, we have added a new model named Mixed Weighted Gaussian(MWG). Related papers will be submitted online later.
The Structure of the MWG algorithm:
<div align="center">
  <img src="https://github.com/xiaojianyang820/Dopamine/blob/main/images/fig_1.jpg"><br><br>
</div>
The core innovation of this algorithm lies in:
- The use of a Gaussian distribution allows for efficient and accurate sampling operations. An accurate sample-set corresponding to the Gaussian mixture model can be obtained by combining the results of sampling from each Gaussian component.
- The estimation result of statistics from the previous round is used as an initial value for the EM method. By embedding a single-step EM operation into the RL iteration, the stability of the learning process is improved without introducing excessive computational burden.
- MWG leverages the convenience of the sample-set to approximate the expected distribution by concatenating the sample-sets from before and after the Bellman update. This process addresses the issue of the EM method not being an affine transformation.

The core program file of this algorithm is placed in `dopamine/agents/mg` .

#### Learning Curve of different algorithms on 12 games (5 random seeds)
<div align="center">
  <img src="https://github.com/xiaojianyang820/Dopamine/blob/main/images/the learning curve of different algorithms.png"><br><br>
</div>

#### Summary Table of normalized score on all 59 atari games

| - | Mean | Median | Best Count | BTH Count |
|---|------|--------|------------|-----------|
| DQN     | 2.66 | 0.67 | 0  | 20 |
| Rainbow | 5.22 | 1.48 | 7  | 42 |
| IQN     | 5.56 | 1.25 | 5  | 40 |
| MMD     | 4.79 | 1.36 | 11 | 38 |
| MWG     | 8.51 | 2.19 | 36 | 49 |

#### Start Training
You can use the following sample commands to start model training. The `.gin` file is the configuration file of the algorithm parameters.
`base_dir` specifies the storage folder of training logs and model parameters. If the folder is not empty, the program will read the existing parameters first and continue to run the program.
    
    export game_name=asteroids
    python -um dopamine.discrete_domains.train --base_dir tmp/mwg_${game_name} \
        --gin_files dopamine/agents/mg/configs/mwg_${game_name}.gin
	
The pre-trained model file can be downloaded at this [address](https://drive.google.com/drive/folders/1HG2rkYvuisQHmLakWAtRf6J1mY9yRjJ1?usp=sharing), and the downloaded file is placed in the `tmp` folder. Use the above command to reload the neural network parameters and restart the relevant training.

Some test videos of Atari games can be obtained under this [address](https://www.youtube.com/watch?v=HylLIiSdnFA&list=PLLx_dwVwxN9XK36QVFVKTxzqCTXgDF-bE).

