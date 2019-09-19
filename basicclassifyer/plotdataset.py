import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv')

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

# counts number of times each type made a comment
for i in range(len(dataset)):
    typeCount[dataset.loc[i,'type']] = typeCount.get(dataset.loc[i,'type'], 0) + 1

print(typeCount)
# graphs the number of comments by each type
plt.bar(range(len(typeCount)), list(typeCount.values()), align='center')
plt.xticks(range(len(typeCount)), list(typeCount.keys()))
plt.show()
