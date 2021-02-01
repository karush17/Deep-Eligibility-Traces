# Deep Eligibility Traces

<!-- benchmarking-
for each env: rewards (baselines+3 traces) x6 algos, variation lambda (3 traces) x4 values [0, 0.5, 0.75, 1] x6 algos -->

## Introduction
This repository consists of implementations of Eligiblity Traces and corresponding algorithms in the deep learning setting. Algorithms are implemented in [`PyTorch`](Pytorch/) and [`Tensorflow 2.0`](Tensorflow/) on a range of problems. Custom toy problems are provided in the [`MDPs`](MDPs/) folder.

## Baseline Algorithms
Following are the baseline algorithms combined with trace-based updates-

|Algorithm|Link|Implementation|Status|
|:-------:|:--:|:------------:|:---:|
|Sarsa|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`sarsa.py`](Pytorch/sarsa.py)|:heavy_check_mark:|
|Double Sarsa|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`doublesarsa.py`](Pytorch/doublesarsa.py)|:heavy_check_mark:|
|Q-Learning|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`qlearning.py`](Pytorch/qlearning.py)|:heavy_check_mark:|
|Double Q-Learning|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`doubleqlearning.py`](Pytorch/doubleqlearning.py)|:heavy_check_mark:|
|Expected Sarsa|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`expectedsarsa.py`](Pytorch/expectedsarsa.py)|:heavy_check_mark:|
|Double Expected Sarsa|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`doubleexpectedsarsa.py`](Pytorch/doubleexpectedsarsa.py)|:heavy_check_mark:|


## Trace Algorithms
Following algorithms are available in the current version-

### PyTorch

|Trace|Baseline Algorithms|Link|Implementation|Status|
|:---:|:------------------|:--:|:------------:|:----:|
|TD(λ)|TD(1)|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`TDlamb.py`](Pytorch/TDlamb.py)|Requires tuning|
|Replacing Trace|<ul><li>- [x] Sarsa</li><li>- [x] Q-learning</li><li>- [x] Expected Sarsa</li><li>- [x] Double Sarsa</li><li>- [x] Double Q-learning</li><li>- [x] Double Expected Sarsa</li></ul>|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`traces.py`](traces/traces.py)|:heavy_check_mark:|
|Accumulating Trace|<ul><li>- [x] Sarsa</li><li>- [x] Q-learning</li><li>- [x] Expected Sarsa</li><li>- [x] Double Sarsa</li><li>- [x] Double Q-learning</li><li>- [x] Double Expected Sarsa</li></ul>|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`traces.py`](traces/traces.py)|:heavy_check_mark:|
|Dutch Trace|<ul><li>- [x] Sarsa</li><li>- [x] Q-learning</li><li>- [x] Expected Sarsa</li><li>- [x] Double Sarsa</li><li>- [x] Double Q-learning</li><li>- [x] Double Expected Sarsa</li></ul>|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`traces.py`](traces/traces.py)|:heavy_check_mark:|

<!-- ### Tensorflow 2.0
|Algorithm|Link|Implementation|
|:-------:|:--:|:------------:|
|TD-lambda|[Sutton & Barto, Chapter 12](http://incompleteideas.net/book/RLbook2020.pdf)|TBA|
 -->

## Custom Environments
Following is the list of custom toy environments-

|Environment Name|Link|Implementation|
|:--------------:|:--:|:------------:|
|CyclicMDP|[ESAC](https://arxiv.org/pdf/2007.13690.pdf)|[link](MDPs/cyclic_mdp.py)|
|OneStateMDP|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[link](MDPs/one_state_mdp.py)|
|OneStateGaussianMDP|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[link](MDPs/one_state_gaussian_mdp.py)|
|GeneralizedCyclicMDP|motivated by [ESAC](https://arxiv.org/pdf/2007.13690.pdf)|[link](MDPs/gen_cyclic_mdp.py)|
|StochasticMDP|[hDQN](https://arxiv.org/pdf/1604.06057.pdf)|[link](MDPs/stochastic_mdp.py)|
|MultiChainMDP|[ET(λ)](https://arxiv.org/pdf/2007.01839.pdf)|[link](MDPs/multi_chain_mdp.py)|

## Usage

To run an implementation, use the following command- 
```
python main.py --configs configs/configs.yaml --log_dir log/ --env <ENVIRONMENT>
```

For example, to run Q-Learning on the CyclicMDP environment using PyTorch library, use the following-
```
python main.py --configs configs/configs.yaml --log_dir log/ --env CyclicMDP
```
This will train the agent with default arguments listed in [`configs.yaml`](configs/configs.yaml) file. 

Following is an example to enter custom arguments for Q-Learning on the CartPole-v0 environment using PyTorch library with replacing trace and lambda=0.5-
```
python main.py --configs configs/configs.yaml --log_dir log/ --alg QLearning --env CartPole-v0 --lib torch --trace replacing --lamb 0.5 --num_steps 10000
``` 

## Citation
If you find these implementations helpful then please cite the following-
```
@misc{karush17eligibilitytraces,
  author = {Karush Suri},
  title = {Deep Eligibility Traces},
  year = {2021},
  howpublished = {\url{https://github.com/karush17/Deep-Eligibility-Traces}},
  note = {commit xxxxxxx}
}
```


