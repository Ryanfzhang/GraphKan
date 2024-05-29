# Graphkan: Implementation of Graph Neural Network version of Kolmogorov Arnold Networks with torch geometrics

In this repository, we have implemented Graphkan using pytorch_geometric. Therefore, you can easily apply GraphKan to your own graph tasks. We have tested GraphKan on signal classification tasks, and the results are as follows.

| Model     | Set 1 | Set 2 | Set 3 | Set 4 |
| ----------- | ----------- |----------- |----------- |----------- |
| GCN      | 0.9343       |0.9457|0.8871|0.8186|
| GCN with  Kolmogorov Arnold Networks  |  0.9643       |0.9600 |0.9214|0.8400|

Learning curve on S1 is as below.

![image](https://github.com/Ryanfzhang/GraphKan/assets/150044070/de152261-7ee6-4891-b37b-a1ac7109d549)

Training log is uploaded as PDF.

GCN: \log\graph.pdf 

GCN with  Kolmogorov Arnold Networks: \log\graphkan.pdf
## Usage
If you want to use GraphKan on your own task, you can replace original ChebConv with the kanChebConv in model/GNNs.py. 

## Tips
(1) You can utilize various types of KAN, including [FourierKan](https://github.com/GistNoesis/FourierKAN), [ChebyKan](https://github.com/SynodicMonth/ChebyKAN) and so on. However, it is unfortunate that FourierKAN and ChebyKAN are not effective for our tasks, and further investigation is needed to determine the cause.

(2) According to our experiments, the inclusion of LayerNorm or BatchNorm in your network may be necessary.

Thanks to the [efficient-kan](https://github.com/Blealtan/efficient-kan/).
