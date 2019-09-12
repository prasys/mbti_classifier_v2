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

submission = reddit.submission(url='https://www.reddit.com/r/mbti/comments/c43fl8/maybe_the_most_infp_thing_ive_ever_seen/')

submission.author_flair_text

count = 0
df = pd.DataFrame()

submission.comments.replace_more(limit=None)
for comment in submission.comments.list():
    df.loc[count,'comment'] = comment.body
    df.loc[count,'type'] = comment.author_flair_text
    count += 1

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
    'None' : 0,
}

for i in range(len(df)):
    typeCount[df.loc[i,'type']] = typeCount.get(df.loc[i,'type'], 0) + 1

plt.bar(range(len(typeCount)), list(typeCount.values()), align='center')
plt.xticks(range(len(typeCount)), list(typeCount.keys()))
plt.show()
# print(typeCount)
