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

submission = reddit.submission(url='https://www.reddit.com/r/mbti/comments/d2uay0/looking_at_you_xnxps/')

submission = reddit.submission(id='d2uay0')

submission.author_flair_text

count = 0
df = pd.DataFrame()

for top_level_comment in submission.comments:
    if isinstance(top_level_comment, MoreComments):
        continue
    df.loc[count,'comment'] = top_level_comment.body
    df.loc[count,'type'] = top_level_comment.author_flair_text
    count += 1

print(df.head())
