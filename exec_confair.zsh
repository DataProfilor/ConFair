# exec this script to run the experiments for ConFair and DifFair 
# in their performance over real-world data
# The execution of this script requires the python 3.7 and packages listed in requirements.txt.
# And the execution of this script need to be inside the local folder "ConFair".

python PrepareData.py
python RepairDataByCAP.py # special heck for CAP

python LearnCCrules.py
python TrainMLModels.py
python TrainMLModelswithReweigh.py --weight 'scc' --base 'kam'
python TrainMLModelswithReweigh.py --weight 'omn' --base 'one'
python TrainMLModelswithReweigh.py --weight 'kam' --base 'one'
python TrainMLModelswithReweigh.py --weight 'cap' --base 'one'
python EvaluateModels.py 
# produce "res-X.csv" for each dataset X of real data
python SummarizeEvaluations.py

# extract run-time of all methods, produce "time-X.csv" for each dataset X of real data
python SummarizeTime.py