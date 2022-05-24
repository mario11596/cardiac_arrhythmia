# Cardiac arrhythmia prediction based on machine learning

## About
Today, arrhythmia is one of the diseases that can be easily diagnosed, but also successfully treated with the necessary medication. I have used five algorithms in the programming language Python, of which the best results were with SVC and Gradient boost. The main problem with machine learning in this dataset is mostly uneven data and too much dominance of one class. The algorithms used are as follows: Decision Trees, Random Forest, Gradient Boost, Support Vector Machine (C-Support Vector) and K-Nearest Neighbour. I tried to show that we can speed up the process of detecting diseases in medicine, with an important prerequisite, which is a large set of data with different classifications.

## Dataset
The dataset that I used is from the website [UCI Machine learning repository](https://archive.ics.uci.edu/ml/datasets/arrhythmia). It consists of 452 instances and 279 attributes with 16 classes as target.

| Class code | Class                                      | Number of instances |
|---------|--------------------------------------------|---------------------|
| 01 | Normal	| 245  |
| 02 | Ischemic changes (Coronary Artery Disease) | 44 |
| 03 | Old Anterior Myocardial Infarction | 15 |
| 04 | Old Inferior Myocardial Infarction | 15 |
| 05 | Sinus tachycardy| 13 |
| 06 | Sinus bradycardy| 25  |
| 07 | Ventricular Premature Contraction (PVC) | 3 |
| 08 | Supraventricular Premature Contraction | 2  |
| 09 | Left bundle branch block  | 9  |
| 10 | Right bundle branch block | 50 |
| 11 | 1. degree AtrioVentricular block | 0 |
| 12 | 2. degree AV block | 0 |
| 13 | 3. degree AV block | 0 |
| 14 | Left ventricule hypertrophy | 4 |
| 15 | Atrial Fibrillation or Flutter | 5 |
| 16 | Others	| 22 |