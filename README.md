# Graphkan: Implementation of Graph Neural Network version of Kolmogorov Arnold Networks with torch geometrics and its application on signal classification

We implement the Graphkan with pytorch_geometric. Hence, you can easily utilize the GraphKan

If the gpu (cuda) running fails, change to cpu training by changing:

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ------> args.device = torch.device('cpu')

Thanks to the original implementations KAN (https://github.com/KindXiaoming/pykan) and FourierKAN (https://github.com/GistNoesis/FourierKAN), you guys are amazing.

Still at experimental stage to see if KAN really works on graph-structured data.
