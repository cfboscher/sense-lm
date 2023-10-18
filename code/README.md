# SENSE-LM 
The code of "SENSE-LM : A Synergy of Language Models and Sensorimotor Representations for Sensory Information Extraction".


## Setup

First create and activate the conda environment as follows : 
```
conda env create -f environment.yml
conda activate sense-lm
```

Then, download the required models: 

```
python -m spacy download en_core_web_md

cd step_2
bash download_roberta.sh
```

If  facing issues with the downloading of roberta, try to clone the repo with `git-lfs` instead, and rename the cloned files properly, as indicated in `step_2/download_roberta.sh`.




Finally, download the Odeuropa Dataset as follows :

```
cd ../../data/
bash download_data.sh
```
## Usage
### Run Experiments

SENSE-LM can be executed as follows : 
```python run.py```

The default configuration is stored in `config/config.py`.
It contains the experimental parameters such as the `dataset`, that must be either set to `"odeuropa"` or `"auditory"`.

Then, according to the steps of SENSE-LM you want to run, the attribute `steps` as `"all"`, `"1"` or `"2"` (`"all"` by default).

Other parameters are explicitely named experimental values.

### Run experiments with custom configuration

Custom configurations are provided in folder `config/config_files`. Each parameter defined in the JSON file overrides the default attribute in `config/config.py`.

Non-specified attributes are assigned the default value.

For instance, to run SENSE-LM with the custom configuration `config/config_files/odeuropa.json`, execute the following command : 

``` python run.py --config_file=config/config_files/odeuropa.json```.

The config files must be structured as follows, based on the example of  `odeuropa.json`: 
```
{
  "config":
  {

        "device" : "cuda" ,
        "data_path" : "../data" ,
        "dataset" : "odeuropa" ,
        "random_seed" :42,
    
        "steps" :"2",


        "pretrained_parameters_step1" :"emanjavacas/MacBERTh",
        "epochs_step1" :20,
        "learning_rate_step1" :2e-5,
        "epsilon_step1" :1e-8,
    
        "epochs_step2" :20,
        "learning_rate_step2" :2e-5,
        "tokenizer_max_len_step2" :303,
        "epsilon_step2" :1e-8,
        "T" :3.5,
        "U" :0.95
  }
}
```


## Code Artifacts

This repository includes code artifacts from the following external projects : 

- BERT for sentiment classification using PyTorch : https://github.com/Taaniya/bert-for-sentiment-classification-pytorch
- Extracting Phrase from Sentence : https://github.com/Jitendra-Dash/Extracting-Phrase-From-Sentence
- Sensorimotor Distance Calculator : https://github.com/emcoglab/sensorimotor-distance-calculator

- BERT : https://huggingface.co/bert-base-uncased
- MacBERTh : https://huggingface.co/emanjavacas/MacBERTh
- RoBERTa : https://huggingface.co/roberta-base


