# Graphkan: Implementation of Graph Neural Network version of Kolmogorov Arnold Networks with torch geometrics

In this repository, we implement the Graphkan with pytorch_geometric. Hence, you can easily utilize the GraphKan on your own graph task. And we test our GraphKan on signal classification tasks, results are as below.

| Model     | Set 1 | Set 2 | Set 3 | Set 4 |
| ----------- | ----------- |----------- |----------- |----------- |
| GCN      | 0.9343       |0.9457|0.8871|0.8186|
| GCN with  Kolmogorov Arnold Networks  |  0.9643       |0.9600 |0.9214|0.8400|
