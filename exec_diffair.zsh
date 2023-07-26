# exec this script to run the experiments for ConFair and DifFair 
# in their performance over synthetic data
# The execution of this script requires the python 3.7 and packages listed in requirements.txt.
# And the execution of this script need to be inside the local folder "ConFair".

python PrepareData.py --data 'all_syn' --exec_n 5
python LearnCCrules.py --data 'all_syn' --exec_n 5
python TrainMLModels.py --data 'all_syn' --model 'lr' --exec_n 5
python TrainMLModelswithReweigh.py --data 'all_syn' --model 'lr' --weight 'scc' --base 'kam' --exec_n 5
python EvaluateModels.py --data 'all_syn' --model 'lr' --exec_n 5
# produce "res-X.csv" for each dataset X of synthetic data
python SummarizeEvaluations.py --data 'all_syn' --model 'lr' --exec_n 5

