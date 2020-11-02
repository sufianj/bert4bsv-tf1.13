# Utiliser BERT pour classifer les phrases extraites des BSV (multi-class, not multi-label)

### Tuto utilisé:  
https://lesdieuxducode.com/blog/2019/4/bert--le-transformer-model-qui-sentraine-et-qui-represente

### Env: 
 - Python 3.7

 - Tensorflow 1.13

 - Version de BERT : https://github.com/google-research/bert/tree/bee6030e31e42a9394ac567da170a89a98d2062f

 - Version du modèle de langue pré-entraîné : https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip


J'ai téléchargé Bert ici: https://github.com/google-research/bert , pour faire fonctionner les codes, les models *ckpt* sont manquants car trop gros pour github.

### Codes adaptés/ ajoutés:
load_data.py
run_classifier.py

### jeux de données:
data/stcs_type_50_50.csv
data/stcs_type_50_50_test.csv

### résultats
output/test_results_heure_date.xlsm

#### Customisation du modèle
    python run_classifier.py --task_name=cola --do_train=true --do_eval=true --do_predict=true --data_dir=./data/ --vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./multi_cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=160 --train_batch_size=4 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./output/ --do_lower_case=False

#### Utiliser le classifieur que vous venez d'entraîner
    python run_classifier.py --task_name=cola --do_predict=true --data_dir=./data
    --vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt
    --bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json
    --init_checkpoint=./output/model.ckpt-3000 --max_seq_length=160
    --output_dir=./output/
