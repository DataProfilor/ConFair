python3 PrepareData.py
python3 LearnCCrules.py --exec_n 1
python3 TrainMLModels.py --exec_n 1
python3 RetrainModelwithWeights.py --weight 'scc' --base 'kam' --exec_n 1
python3 RetrainModelwithWeights.py --weight 'kam' --exec_n 1
python3 RetrainModelwithWeights.py --weight 'omn' --exec_n 1
python3 EvaluateModels.py --exec_n 1
python3 SummarizeEvaluations.py --exec_n 1