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
Then download the code repository and cd to your downloaded local directory. Run one single execution of the experiments using the below command.

```bash
./ exec.zsh
```

Note that for MEPS16 dataset, you need to extract the raw data using the R scrip. See details at https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/raw/meps/README.md.

Note for running CAPUCHIN, you need to install the CAPUCHIN package.