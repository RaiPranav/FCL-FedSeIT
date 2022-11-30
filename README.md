## FedSeIT: Federated Selective Inter-Client Transfer

![Federated Selective Inter-client Transfer (FedSeIT)](./docs/FedSeIT.png?raw=true "Federated Selective Inter-client Transfer (FedSeIT)")

This code repository contains the implementations of: (a) "**Federated Selective Inter-client Transfer (FedSeIT)**" framework, and (b) "**Selective Inter-client Transfer (SIT)**" method proposed in the paper titled: **["Federated Continual Learning for Text Classification via Selective Inter-client Transfer"](https://arxiv.org/abs/2210.06101)** accepted at **EMNLP2022** Conference in "Findings in EMNLP" track.

### Highlights of the FedSeIT paper

1. **The authors propose a novel framework FedSeIT** which performs domain alignment of foreign clients' task-specific parameters to minimize inter-client interference.
1. **The authors propose a novel task selection method SIT** which efficiently selects relevant parameters from foreign clients to maximize inter-client knowledge transfer.
1. It is the **first work that applies FCL method in NLP domain**.
1. **State-of-the-art performance on Text Classification** task over 5 NLP datasets from diverse domains.


### Environment Setup
Download 300-dimensional "**glove.840B.300d.txt**" GloVe embeddings from [this](https://nlp.stanford.edu/projects/glove/) link and place it in Resources/ directory.

```
> python3.8 -m venv venv
> source venv/bin/activate
> pip3 install -r requirements.txt
```

### FCL Data Generation and Preprocessing

This bash script below needs to be run only once, to create tasks from the dataset in a random order (with fixed random seed for reproducibility).
Here we have already provided the preprocessed data for Reuters8 (R8) dataset.
However, the bash scripts are also provided for TMN and TREC6 datasets.

```
> cd Config/r8/
> bash data.sh
```

### Run CNN Text Classification model with FedWeIT framework

All configurable parameters for model training in FCL setting can be found in FedWeIT/parser.py file.

```
> cd Config/r8/
> bash FedWeIT.sh
```

### Run CNN Text Classification model with FedSeIT framework
```
> cd Config/r8/
> bash FedSeIT.sh
```

### How to cite this work

```
@misc{https://doi.org/10.48550/arxiv.2210.06101,
  title = {Federated Continual Learning for Text Classification via Selective Inter-client Transfer},
  author = {Chaudhary, Yatin and Rai, Pranav and Schubert, Matthias and Schütze, Hinrich and Gupta, Pankaj},
  doi = {10.48550/ARXIV.2210.06101},
  url = {https://arxiv.org/abs/2210.06101},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```


### Existing FCL Work: FedWeIT

This code is based on the implementation of FedWeIT proposed by:
> Yoon, Jaehong, Wonyong Jeong, Giwoong Lee, Eunho Yang, and Sung Ju Hwang. "Federated continual learning with weighted inter-client transfer." In International Conference on Machine Learning, pp. 12073-12086. PMLR, 2021.

Here, we have made a large number of significant contributions to the FedWeIT code to implement our proposed **FedSeIT** framework and **SIT** method, as cited in our research literature.