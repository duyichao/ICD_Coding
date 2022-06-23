This repo mainly requires the following packages.
* numpy                     1.15.1
* python                    3.6.7
* pytorch                   1.5.0
* allennlp                  0.8.4

Data
-----


Usage
-----
1. Train and test using full MIMIC-III data
  ```
  python main.py -data_path ./data/mimic3/train_full_4_level.json --vocab ./data/mimic3/vocab.csv --Y full --model HMRCNN --criterion f1_micro --level 3 --tune_wordemb --embed_file ./data/mimic3/processed_full.embed --gpu 3
  ```
2. Train and test using top-50 MIMIC-III data
  ```
  python main.py -data_path ./data/mimic3/train_50_4_level.json --vocab ./data/mimic3/vocab.csv --Y 50 --model HMRCNN --criterion f1_micro -level 3 --tune_wordemb -embed_file ./data/mimic3/processed_full.embed --gpu 3
  ```

Optimizer
-----
1. top-50: AdaBelief or AdamW
2. full: AdamW

Word Embedding
-----
