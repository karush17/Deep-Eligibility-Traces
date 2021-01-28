# Deep Eligibility Traces

## Introduction
This repository consists of implementations of Eligiblity Traces and corresponding algorithms in the deep learning setting. Algorithms are implemented in [`PyTorch`](Pytorch/) and [`Tensorflow 2.0`](Tensorflow/) on a range of problems. Custom toy problems are provided in the [`MDPs`](MDPs/) folder.

## Baseline Algorithms
Following are the baseline algorithms combined with trace-based updates-

|Algorithm|Link|Implementation|Status|
|:-------:|:--:|:------------:|:---:|
|Sarsa|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`sarsa.py`](Pytorch/sarsa.py)|Works well|
|Double Sarsa|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`doublesarsa.py`](Pytorch/doublesarsa.py)|:heavy_check_mark:|
|Q-Learning|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`qlearning.py`](Pytorch/qlearning.py)|:heavy_check_mark:|
|Double Q-Learning|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`doubleqlearning.py`](Pytorch/doubleqlearning.py)|:heavy_check_mark:|
|Expected Sarsa|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`expectedsarsa.py`](Pytorch/expectedsarsa.py)|:heavy_check_mark:|
|Double Expected Sarsa|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[`doubleexpectedsarsa.py`](Pytorch/doubleexpectedsarsa.py)|:heavy_check_mark:|


## Trace Algorithms
Following algorithms are available in the current version-

### PyTorch

|Algorithm|Link|Implementation|Status|
|:-------:|:--:|:------------:|:---:|
|TD-lambda|[Sutton & Barto, Chapter 12](http://incompleteideas.net/book/RLbook2020.pdf)|[`TDlamb.py`](Pytorch/TDlamb.py)|Requires tuning|

<!-- ### Tensorflow 2.0
|Algorithm|Link|Implementation|
|:-------:|:--:|:------------:|
|TD-lambda|[Sutton & Barto, Chapter 12](http://incompleteideas.net/book/RLbook2020.pdf)|TBA|
 -->

## Custom Environments
Following is the list of custom toy environments-

|Environment Name|Link|Implementation|
|:--------------:|:--:|:------------:|
|Cyclic MDP|[ESAC](https://arxiv.org/pdf/2007.13690.pdf)|[link](MDPs/cyclic_mdp.py)|
|One-state MDP|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[link](MDPs/one_state_mdp.py)|
|One-state Gaussian MDP|[Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)|[link](MDPs/one_state_gaussian_mdp.py)|

## Usage

To run an implementation, use the following command- 
```
python main.py --alg <ALGORITHM> --env <ENV> --lib <LIBRARY> --trace <TRACE> --lamb <LAMBDA> --num_steps <STEPS>
```

For example, to run Q-Learning on the CartPole-v0 environment using PyTorch library with replacing trace and lambda=0.5, use the following-
```
python main.py --alg QLearning --env CartPole-v0 --lib torch --trace replacing --lamb 0.5 --num_steps 10000
``` 

To view the full list of arguments run the following-
```
python main.py --help
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


