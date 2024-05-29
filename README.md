# Graphkan: Implementation of Graph Neural Network version of Kolmogorov Arnold Networks with torch geometrics

In this repository, we implement the Graphkan with pytorch_geometric. Hence, you can easily utilize the GraphKan on your own graph task. And we test our GraphKan on signal classification tasks, results are as below.

| Model     | Set 1 | Set 2 | Set 3 | Set 4 |
| ----------- | ----------- |----------- |----------- |----------- |
| GCN      | 0.9343       |0.9457|0.8871|0.8186|
| GCN with  Kolmogorov Arnold Networks  |  0.9643       |0.9600 |0.9214|0.8400|

## Usage
If you want to use GraphKan on your own task, you can replace original ChebConv with the kanChebConv in model/GNNs.py. 

## Tips
(1) You can use different kinds of KAN, such as [FourierKan](https://github.com/GistNoesis/FourierKAN), [ChebyKan](https://github.com/SynodicMonth/ChebyKAN). Unfortunately, FourierKAN and ChebyKan do not work on our tasks, the reason still needs to be investigated.
(2) LayerNorm or BatchNorm may be neccessary.

Thanks to the [efficient-kan](https://github.com/Blealtan/efficient-kan/).
