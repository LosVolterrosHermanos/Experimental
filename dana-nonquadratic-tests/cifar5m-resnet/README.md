This is an experiment for running DANA on ResNets.

It assumes access to cifar5m:

https://github.com/preetum/cifar5m?tab=readme-ov-file

It assumes all *.npz files have been colocated in one folder.

The ResNet implementation and experiment code are all in flax-cifar5m-resnet-sweep.

No batchnorm is used, and all activations were replaced by normalized activation functions (softSign).

