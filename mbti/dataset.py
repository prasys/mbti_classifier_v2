import praw
from praw.models import MoreComments
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.options.mode.chained_assignment = None

reddit = praw.Reddit(user_agent='Comment Extraction (by /u/mbti_)',
                     client_id='TCvLEOtM48aFJg', client_secret='OzboPnt9dvEbbLTexylSlWb-7kc',
                     username='mbti_', password='0p3np0dd00r')

# this is to append the comments from each top post
dataset = []

# gets the comment forest from each of the n# of top posts
# creates a dataframe of the types as well as their corresponding comment
for submission in reddit.subreddit('mbti').hot(limit=50):
    submission = reddit.submission(id=submission.id)
    count = 0
    df = pd.DataFrame()
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        df.loc[count,'comment'] = comment.body
        df.loc[count,'type'] = comment.author_flair_text
        count += 1
    dataset.append(df)

# cleans the dataframe by removing comments with no myers briggs type
dataset = pd.concat(dataset)
dataset = dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
dataset['num'] = range(len(dataset))
dataset.reset_index(level=0, inplace=True)
dataset.set_index('num')
dataset = dataset[['type','comment']]
dataset.to_csv('dataset.csv')

# all mbti types
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

# graphs the number of comments by each type
plt.bar(range(len(typeCount)), list(typeCount.values()), align='center')
plt.xticks(range(len(typeCount)), list(typeCount.keys()))
plt.show()
