# Classifying malicious network traffic using k-NN

This project uses k-NN to classifiy malcious network traffic in the Intrusion Detection Evaluation Dataset (CICIDS2017) located [here](https://www.unb.ca/cic/datasets/ids-2017.html) .

Specifically, we use the Friday, July 7, 2017 AFTERNOON dataset, which involves
a port scan from Attacker Kali 205.174.165.73 on Victim Ubuntu16 205.174.165.68 (Local IP: 192.168.10.50)

The dataset comes with 80 historically useful features already extracted by the [CICFlowMeter-V3](https://www.unb.ca/cic/research/applications.html) .

## Feature Selection 

From these features, we selected the most dominant using a special technique called
Fast Orthogonal Search (FOS). The use of FOS for this purpose will be the topic of a future publication. 
We have broken the dataset into the following subsets, which are used for training and prediction: 

1. training set = 'data/train_inputs.csv'         
2. training labels = 'data/train_outputs.csv'      
3. test set = 'data/test_inputs.csv'            
4. test labels (for evaluation) = 'data/test_outputs.csv'         

Note: Malicious traffic was labelled with a 1 and benign traffic was labelled with a -1

## Learning

We use the k-NN implementation from the scikit-learn library, which is an ancient technique and well-documented [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) .

Just run *main.py*. It should take about 2 mins, depending on your CPU.

## Results

Output should look something like this:

- k-NN classified with  99.7 %% accuracy
- The training took  69.5 secs
- The classification took  49.8 secs