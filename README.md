# ConFair
This is the repository for DifFair and ConFair. 

## Initialize a virtual environment and install the libraries

```bash
conda create -n 'venv' python=3.7.0
conda activate venv
pip install -r requirements.txt
```

## Install CC tool
Download the folder "DataInsights" from https://github.com/microsoft/prose/tree/main/misc/CCSynth/CC and copy this folder inside your local directory of this repository.
```bash
pip install -e DataInsights
```

## Run the tool
Then download the code repository and cd to your downloaded local directory "ConFair". Run one single execution of the experiments using the below command.

### Execute below script to run the experiments for ConFair and DifFair in their performance over real-world data

```bash
./ exec_confair.zsh
```

### Execute below script to run the experiments for ConFair and DifFair in their performance over synthetic data

```bash
./ exec_diffair.zsh
```

### Execute below script to run the experiments in comparing ConFair to OMN in their performance under model-aware weights

```bash
./ exec_aware.zsh
```

### Execute below script to run the experiments in comparing ConFair and DifFair in their performance without the optimization of CCs over real data

```bash
./ exec_opt_cc.zsh
```

### Execute below script to run the experiments in comparing ConFair to OMN in their relationship between inpute degree and fairness improvement

```bash
./ exec_degree.zsh
```



Note that for MEPS16 dataset, you need to extract the raw data using the R scrip. See details at https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/raw/meps/README.md.

Note for running CAPUCHIN, you need to install the CAPUCHIN package.

## Visualization
Some visualization in the paper can be found at this folder [notebooks](https://github.com/DataProfilor/ConFair/tree/main/notebooks).
