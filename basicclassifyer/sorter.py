import pandas as pd
import pprint
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv')

# mixes rows
dataset = dataset.sample(frac=1).reset_index(drop=True)

typeCount = {
    'ENTJ' : 0,
    'ENTP' : 0,
    'ENFJ' : 0,
    'ENFP' : 0,
    'ESTJ' : 0,
    'ESTP' : 0,
    'ESFJ' : 0,
    'ESFP' : 0,
    'INTJ' : 0,
    'INTP' : 0,
    'INFJ' : 0,
    'INFP' : 0,
    'ISTJ' : 0,
    'ISTP' : 0,
    'ISFJ' : 0,
    'ISFP' : 0,
}

df = pd.DataFrame()

for i in range(len(dataset)):
    if typeCount[dataset.loc[i,'type']] < 30:
        df.loc[i,'type'] = dataset.loc[i,'type']
        df.loc[i,'comment'] = dataset.loc[i,'comment']
        typeCount[dataset.loc[i,'type']] += 1
    else:
        df.loc[i,'type'] = 0

for i in range(len(df)):
    if df.loc[i,'type'] == 0:
        df = df.drop(index=i)

fig = plt.figure(figsize=(8,6))
df.groupby('type')['comment'].count().plot.bar(ylim=0)
plt.show()

df = df.reset_index(drop=True)

df.to_csv('balancedDataset.csv')
