import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
###origine
###df = pd.read_csv("data/train.csv", sep=';', encoding='utf-8') #En cas d'erreur essayez avec d'autres encodings
#mien
df = pd.read_csv("data/stcs_type_50_50.csv", sep=',', encoding='utf-8') #En cas d'erreur essayez avec d'autres encodings

# Crée les DataFrames train et dev dont BERT aura besoin, en ventillant 1 % des données dans test
###df_bert = pd.DataFrame({'user_id':df['ID'], 'label':le.fit_transform(df['Label']), 'alpha':['a']*df.shape[0], 'text':df['texte'].replace(r'\n',' ',regex=True)})
#mien
df_bert = pd.DataFrame({'user_id':0,'label':le.fit_transform(df['label']), 'alpha':'a', 'text':df['report_text'].replace(r'\n',' ',regex=True)})
df_bert.index = [x for x in range(0, len(df_bert))]
df_bert['user_id'] = df_bert.index

df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01) 

# Crée la DataFrame test dont BERT aura besoin
###df_test = pd.read_csv("data/test.csv", sep=';', encoding='utf-8') #En cas d'erreur essayez avec d'autres encodings
#mien
df_test = pd.read_csv("data/stcs_type_50_50_test.csv", sep=',', encoding='utf-8')

###df_bert_test = pd.DataFrame({'user_id':df_test['ID'], 'text':df_test['texte'].replace(r'\n',' ',regex=True)})
#mien
df_bert_test = pd.DataFrame({'user_id':0, 'text':df_test['report_text'].replace(r'\n',' ',regex=True)})
df_bert_test.index = [x for x in range(0, len(df_bert_test))]
df_bert_test['user_id'] = df_bert_test.index

# Enregistre les DataFrames au format .tsv (tab separated values) comme BERT en a besoin
df_bert_train.to_csv('data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/test.tsv', sep='\t', index=False, header=True)