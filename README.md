# Deep Eligibility Traces

## Introduction
This repository consists of implementations of Eligiblity Traces and corresponding algorithms in the deep learning setting. Algorithms are implemented in [`PyTorch`]('Pytorch/') and [`Tensorflow 2.0`]('Tensorflow/') on a range of problems. Custom toy problems are provided in the [`MDPs`]('MDPs/') folder.

## Trace Algorithms
Following algorithms are available in the current version-

### PyTorch
|Algorithm|Trace|Link|Implementation|
|:-------:|:---:|:--:|:------------:|
|TD-lambda|<ul><li>- [ ] Accumulating Trace</li><li>- [ ] Replacing Trace</li><li>- [ ] Dutch Trace</li></ul>|[Sutton & Barto, Chapter 12](http://incompleteideas.net/book/RLbook2020.pdf)|TBA|

### Tensorflow 2.0
|Algorithm|Trace|Link|Implementation|
|:-------:|:---:|:--:|:------------:|
|TD-lambda|<ul><li>- [ ] Accumulating Trace</li><li>- [ ] Replacing Trace</li><li>- [ ] Dutch Trace</li></ul>|[Sutton & Barto, Chapter 12](http://incompleteideas.net/book/RLbook2020.pdf)|TBA|


## Custom Environments
Following is the list of custom toy environments-

|Environment Name|Link|Implementation|
|:--------------:|:--:|:------------:|
|Cyclic MDP|[ESAC]()|TBA|
|One-state MDP|[Sutton & Barto]()|TBA|
|One-state Gaussian MDP|[Sutton & Barto]()|TBA|

## Usage
Notes on running implementations to be updated soon.

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


