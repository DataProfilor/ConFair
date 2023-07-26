# exec this script to run the experiments in comparing ConFair to OMN 
# in their performance under model-aware weights
# The execution of this script requires the python 3.7 and packages listed in requirements.txt.
# And the execution of this script need to be inside the local folder "ConFair".
# Run after the execution of "exec_confair.zsh"

python RetrainModelAware.py --weight 'scc' --base 'kam'
python RetrainModelAware.py --weight 'omn' --base 'one'
python EvaluateModels.py --setting 'single' --aware 1
# produce "res-aware-X.csv" for each dataset X of real data
python SummarizeEvaluations.py --eval '-aware' 