# Federated Selective Inter-Client Transfer

This code repository contains implementation of the "FedSeIT" model proposed in "Federated Continual Learning for Text Classification via Selective Inter-client Transfer".

This code is based on the implementation of FedWeIT proposed by:
	Yoon, Jaehong, Wonyong Jeong, Giwoong Lee, Eunho Yang, and Sung Ju Hwang. "Federated continual learning with weighted inter-client transfer." In International Conference on Machine Learning, pp. 12073-12086. PMLR, 2021.

However, we have made a large number of modifications to the codebase provided by the authors 
of FedWeIT, as cited in our research literature. Our modifications mostly revolved
around (1) FedSeIT etc functionality not present in original work (2) Elimination of 
certain FedWeIT and FedWeIT legacy code which were not pertinent (3) Readability and 
Flexibility improvements, as well as Documentation.

## Setup
```
 python3.8 -m venv venv
 source venv/bin/activate
 pip3 install -r requirements.txt
```

Download 'glove.840B.300d.txt' and place it in Resources/


## Data Generation 

This needs to be called only once, to create tasks from the dataset
```
cd Config/r8/
bash data.sh
```

## Run FedWeIT setting
```
cd Config/r8/
bash FedWeIT.sh
```

## Run FedSeIT setting
```
cd Config/r8/
bash FedSeIT.sh
```
Note that our codebase currently refers to FedSeIT as FedPrIT

# Other Notes

A discussion of important parameters and their meaning can be found in FedWeIT/parser.py

