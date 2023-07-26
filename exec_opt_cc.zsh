# exec this script to run the experiments for ConFair and DifFair 
# in their performance without the optimization of CCs over real data
# The execution of this script requires the python 3.7 and packages listed in requirements.txt.
# And the execution of this script need to be inside the local folder "ConFair".

python LearnCCrules.py --opt 0
python TrainMLModelswithReweigh.py --weight 'scc' --base 'kam' --opt 0
python EvaluateModels.py --opt 0
# produce "res-noOPT-X.csv" for each dataset X of real data
python SummarizeEvaluations.py --eval '-noOPT'