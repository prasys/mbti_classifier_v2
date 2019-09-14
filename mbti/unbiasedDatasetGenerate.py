# obtains data from users other commments (not on myers any of the blacklisted psychology subreddits)
import praw
from praw.models import MoreComments
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.options.mode.chained_assignment = None

reddit = praw.Reddit(user_agent='Comment Extraction (by /u/mbti_)',
                     client_id='TCvLEOtM48aFJg', client_secret='OzboPnt9dvEbbLTexylSlWb-7kc',
                     username='mbti_', password='0p3np0dd00r')

typeCountUpper = [
    'ENTJ',
    'ENTP',
    'ENFJ',
    'ENFP',
    'ESTJ',
    'ESTP',
    'ESFJ',
    'ESFP',
    'INTJ',
    'INTP',
    'INFJ',
    'INFP',
    'ISTJ',
    'ISTP',
    'ISFJ',
    'ISFP'
]

typeCountLower = []

for word in typeCountUpper:
    typeCountLower.append(word.lower())
blacklist = ['mbti','mbtimemes','shittyMBTI'] + typeCountLower + typeCountUpper

count = 0
dataset = pd.DataFrame()
authors = []

number_of_posts = 0

for submission in reddit.subreddit('mbti').all(limit=1000):
    number_of_posts += 1
    submission = reddit.submission(id=submission.id)
    df = pd.DataFrame()
    submission.comments.replace_more(limit=None)
    for comment in submission.comments:
        numComments = 0
        if str(comment.author) in authors:
            break
        if comment.author is None:
            break
        for subcomment in comment.author.comments.top('all'):
            # Line checks if the author flair is one of the 16 types.
            # If it is not then that author gets thrown out
            if comment.author_flair_text not in typeCountUpper:
                break
            if numComments > 10:
                break
            if subcomment.subreddit.display_name in blacklist:
                continue
            if len(subcomment.body) < 30:
                continue
            dataset.loc[count,'author'] = str(comment.author)
            dataset.loc[count,'subreddit'] = str(subcomment.subreddit.display_name)
            dataset.loc[count,'type'] = str(comment.author_flair_text)
            dataset.loc[count,'comment'] = str(subcomment.body)
            authors.append(str(comment.author))
            numComments += 1
            count += 1
            print("Posts: " + str(number_of_posts) + " Current Author: " + str(comment.author) + " Total comments: " + str(count))

dataset.to_csv('dataset.csv')
