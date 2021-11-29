# DialogRE-baselines
This repository contains several simple yet strong baselines for dialogue relation extraction task.
The models are implemented based on the baseline model of paper "Semantic Representation for Dialogue Modeling".
You may find the paper [here](https://arxiv.org/pdf/2105.10188).

# Requirements
+ python 3.8
+ pytorch 1.8
+ Tesla V100 
+ transformers 4.8.2


# Preprocessing
Preprocess the data using BERT or Roberta tokenizer.
```
bash /path/to/code/preprocess.sh /path/to/bert-base-uncased
bash /path/to/code/preprocess.sh /path/to/roberta-base
```
It should be noted that we use the BERT model [here](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip), which is different from the `bert-base-uncased` model provided by Huggingface.
The BERT model can be obtained by running:
```
bash utils/convert.sh
```
# Training
```
bash run-bertc-base.sh workplace/data-v2-bert-bin/ train          # BERT base model
bash run-robertac-base.sh workplace/data-v2-roberta-bin/ train    # Roberta base model
bash run-robertac-large.sh workplace/data-v2-roberta-bin/ train   # Roberta large model
```

# Evaluation
```
bash run-bertc-base.sh workplace/data-v2-bert-bin/ test
bash eval.sh  /path/to/save_path/
```

# Pretrained Models

## Todo


# Reference
```
@inproceedings{bai-etal-2021-semantic,
    title = "Semantic Representation for Dialogue Modeling",
    author = "Bai, Xuefeng  and
      Chen, Yulong  and
      Song, Linfeng  and
      Zhang, Yue",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.342",
    doi = "10.18653/v1/2021.acl-long.342",
    pages = "4430--4445"
}
```
