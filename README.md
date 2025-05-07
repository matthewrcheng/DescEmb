# Reproducing the Results of Unifying Heterogeneous Electronic Health Record Systems via Clinical Text-Based Code Embedding
Suman Patra and Matthew Cheng

This repository provides official Pytorch code to implement DescEmb, a code-agnostic EHR predictive model.

The paper can be found in this link:
[Unifying Heterogeneous Electronic Health Record Systems via Clinical Text-Based Code Embedding](https://arxiv.org/abs/2108.03625)

# Requirements

* [PyTorch](http://pytorch.org/) version >= 1.8.1
* Python version >= 3.7, <=3.10

# Steps Taken
## Prepare training data
First, download the datasets from these links (make sure you have Physionet access): 

[MIMIC-III](https://physionet.org/content/iii/1.4/)

[eICU](https://physionet.org/content/eicu-crd/2.0/)

[ccs_multi_dx_tool_2015](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip)

[icd10cmtoicd9gem](https://data.nber.org/gem/icd10cmtoicd9gem.csv)

Second, make directory structure like below:
```
data_input_path
├─ eicu
│  ├─ diagnosis.csv
│  ├─ infusionDrug.csv
│  ├─ lab.csv
│  ├─ medication.csv
│  └─ patient.csv
├─ mimic
│  ├─ ADMISSIONS.csv
│  ├─ D_ICD_PROCEDURES.csv
│  ├─ D_ITEMS.csv
│  ├─ D_LABITEMBS.csv
│  ├─ DIAGNOSES_ICD.csv
│  ├─ ICUSTAYS.csv
│  ├─ INPUTEVENTS_CV.csv
│  ├─ INPUTEVENTS_MV.csv
│  ├─ LABEVENTS.csv
│  ├─ PATIENTS.csv
│  ├─ PRESCRIPTIONS.csv
│  └─ PROCEDURES.csv
├─ ccs_multi_dx_tool_2015.csv
└─ icd10cmtoicd9gem.csv

```
```
data_output_path
├─mimic
├─eicu
├─pooled
├─label
└─fold
```
Then run preprocessing code
```shell script
$ python preprocess_main.py
    --src_data $data
    --dataset_path $data_src_directory
    --dest_path $run_ready_directory 
```
where src_data is either eicu or mimiciii, dataset_path is the input path, and dest_path is the output path. Optionally, you can also include the --ccs_dx_tool_path.
Note that pre-processing takes about 1hours in 128 cores of AMD EPYC 7502 32-Core Processor, and requires more than 16GB of RAM.

## Pre-training a model
### Pre-train a DescEmb model with Masked Language Modeling (MLM)

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --src_data $data \
    --task mlm \
    --mlm_prob $percent \
    --model $model
```

### Pre-train a CodeEmb model with Word2Vec

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --src_data $data \
    --task w2v \
    --model codeemb
```
Note that `--input-path ` should be the root directory containing preprocessed data.

$data should be set to 'mimic' or 'eicu'

$percent should be set to probability (default: 0.3) of masking for MLM

$model should be set to 'descemb_bert' or 'descemb_rnn'

## Training a new model
Below are configurations for the training that can be set. Other hyperparameters can be set as well, but these are defaulting to the values used in the paper.

`$descemb` should be 'descemb_bert' or 'descemb_rnn'

`$ratio` should be set to one of [10, 30, 50, 70, 100] (default: 100)

`$value` should be set to one of ['NV', 'VA', 'DSVA', 'DSVA_DPE', 'VC']

`$task` should be set to one of ['readmission', 'mortality', 'los_3day', 'los_7day', 'diagnosis']

Note that `--input-path ` should be the root directory containing preprocessed data.
### Train a new CodeEmb model:

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task
```
### Train a new DescEmb model:

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model ehr_model \
    --embed_model $descemb \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task
```
For our training, we set the following values:
--value-mode DSVA_DPE
--task mortality
--batch_size 16

Note: if you want to train with pre-trained BERT model, add command line parameters `--init_bert_params` or `--init_bert_params_with_freeze`. `--init_bert_params_with_freeze` enables the model to load and freeze BERT parameters.

## Fine-tune a pre-trained model

### Fine-tune a pre-trained CodeEmb model:

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model_path /path/to/model.pt \
    --load_pretrained_weights \
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task
```
### Fine-tune a pre-trained DescEmb model:
```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model_path /path/to/model.pt \
    --load_pretrained_weights \
    --model ehr_model \
    --embed_model $descemb \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task
```

## Transfer a trained model
```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model_path /path/to/model.pt \
    --transfer \
    --model ehr_model \
    --embed_model $embed_model \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task \
```
Note that `--embed_model` and `pred_model` should be matched with the transferred model. The input path remains the root directory containing preprocessed data. To get the model path, find the most recent checkpoint for the target input model under /outputs/

In our project, we performed the above steps for both DescEmb and CodeEmb using both MIMIC-III and eICU as datasets. We also tried transfer learning between both datasets.

# License
This repository is MIT-lincensed.

# Original Paper Citation
```
@misc{hur2021unifying,
      title={Unifying Heterogenous Electronic Health Records Systems via Text-Based Code Embedding}, 
      author={Kyunghoon Hur and Jiyoung Lee and Jungwoo Oh and Wesley Price and Young-Hak Kim and Edward Choi},
      year={2021},
      eprint={2108.03625},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
