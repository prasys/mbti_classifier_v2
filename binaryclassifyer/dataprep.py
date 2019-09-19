import matplotlib.pyplot as plt
import pandas as pd

# read in big dataset of 5000+ entries of each type
dataset = pd.read_csv('../basicclassifyer/dataset.csv')

# create a copy to work with
data = dataset

# loop through the set and label each comment as either an observer or a decider
# see video for definition of observer / decider: https://www.youtube.com/watch?v=V6_QfwX8Q8U

#################################
## OBSERVERS ARE 1 DECIDERS ARE 0
#################################

train_df = pd.DataFrame()
test_df = pd.DataFrame()

train_index = 0
test_index = 0
observerCount = 0
deciderCount = 0

# splits the dataset into equal numbers of observers and deciders for the training set
# and puts the rest of the data into the testing dataset
for i in range(len(data)):
    type = data.loc[i,'type']
    if (type[0] == 'E' and type[3] == 'J') or (type[0] == 'I' and type[3] == 'P'):
        data.loc[i,'classification'] = 'decider'
        if observerCount <= 2099:
            train_df.loc[train_index,'type'] = data.loc[i,'type']
            train_df.loc[train_index,'classification'] = data.loc[i,'classification']
            train_df.loc[train_index,'comment'] = data.loc[i,'comment']
            train_index += 1
        else:
            test_df.loc[test_index,'type'] = data.loc[i,'type']
            test_df.loc[test_index,'classification'] = data.loc[i,'classification']
            test_df.loc[test_index,'comment'] = data.loc[i,'comment']
            test_index += 1
        observerCount += 1
    else:
        data.loc[i,'classification'] = 'observer'
        if deciderCount <= 2099:
            train_df.loc[train_index,'type'] = data.loc[i,'type']
            train_df.loc[train_index,'classification'] = data.loc[i,'classification']
            train_df.loc[train_index,'comment'] = data.loc[i,'comment']
            train_index += 1
        else:
            test_df.loc[test_index,'type'] = data.loc[i,'type']
            test_df.loc[test_index,'classification'] = data.loc[i,'classification']
            test_df.loc[test_index,'comment'] = data.loc[i,'comment']
            test_index += 1
        deciderCount += 1

train_df.to_csv('train_df.csv')
test_df.to_csv('test_df.csv')

o = 0
d = 0
for i in range(len(train_df)):
    if str(train_df.loc[i,'classification']) == 'observer':
        o += 1
    elif str(train_df.loc[i,'classification']) == 'decider':
        d += 1

o2 = 0
d2 = 0
for i in range(len(test_df)):
    if str(test_df.loc[i,'classification']) == 'observer':
        o2 += 1
    elif str(test_df.loc[i,'classification']) == 'decider':
        d2 += 1

print("Training data || observers: " + str(o) + " deciders: " + str(d))
print("Testing data || observers: " + str(o2) + " deciders: " + str(d2))
