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
        data.loc[i,'classification'] = 0
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
        data.loc[i,'classification'] = 1
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

print("Training data || observers: " + str(train_df['classification'].sum()) + " deciders: " + str(len(train_df)-train_df['classification'].sum()))
print("Testing data || observers: " + str(test_df['classification'].sum()) + " deciders: " + str(len(test_df)-test_df['classification'].sum()))
