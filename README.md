# Test Case-Informed Knowledge Tracing for Open-ended Coding Tasks (TIKOC)
This repository contains the code for the paper <a href="https://arxiv.org/abs/2410.10829">Test Case-Informed Knowledge Tracing for Open-ended Coding Tasks</a>. The primary contributions here include 1. the dataset collected from CodeWorkout for proposed GranularKT task, 2. the code for our multitask model, TIKTOC, which both predict students code and binary score on test case level, and 3. the code for running deep KT on granular level KT and updated OKT as baseline. 

## Setup

### Data
We use [CSEDM](https://sites.google.com/ncsu.edu/csedm-dc-2021/) dataset. There are 50 questions in the original dataset and 17 questions has complete test case input and output. To get the test case level correctness, we run students submitted codes through compiler and compared the results with the key solution. We have set a 30 seconds timeout to handle the infinite loop cases. To validate the correctness for each question, we keep the samples that the mean score over all test cases matches the raw score. There are two files in the `data` folder, the dataset_granular_1st.zip file is used for training all models with the students' first submission to all problems and the dataset_granular_all.zip file is used for training models with students' all submission. 

### Environment
We used Python 3.8.18 in the development of this work. Run the following to set up a Conda environment:
```
conda create --name <env> --file requirements.txt
```

## Train and Evaluate KT models
Each of the following runs the script to train a model from the beginning. All parameters can be changed in the 
configs_okt.yaml and configs_granulardkt.yaml file:
```
python baseline_granularDKT.py         # GranularDKT
python main_okt.py                     # Multitask Model
```

With trained model, run the following command for evaluation: 
```
python eval.py
```


#### Best Hyperparameters Found

Multitask Model:
- lstm_lr=5e-5
- lr=1e-5
- multitask_pred_linear=1e-4
- lr_linear=5e-3

