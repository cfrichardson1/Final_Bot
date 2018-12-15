import matplotlib
matplotlib.use('Agg')

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import six
import tweepy

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os
consumer_key = os.environ.get("consumer_key")
consumer_secret = os.environ.get("consumer_secret")
access_token = os.environ.get("access_token")
access_token_secret = os.environ.get("access_token_secret")


# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Settings
me = 'cfrichardson1'

##########################################################
# Added Bot Functions
##########################################################

def df_creator(tweets):
    # WANTED List to hold wanted information from tweets
    date = []
    favorite_count = []
    name = []
    retweet = []
    text = []
    id_ = []

    # Loop through the list of tweets to grab needed info
    for tweet in tweets:
    date.append(tweet['created_at'])
    favorite_count.append(tweet['favorite_count'])
    retweet.append(tweet['retweet_count'])
    text.append(tweet['full_text'])
    id_.append(tweet['id'])

    # Create DF based on WANTED lists
    df = pd.DataFrame({
      'Created': date,
      'Likes': favorite_count,
      'Retweet': retweet,
      'Text': text,
      'ID': id_,
    })

    # Convert date to datetime dtype
    df['Created'] = [datetime.strptime(date, "%a %b %d %H:%M:%S %z %Y") for date in df['Created']]

    return df

# Analyze Pulled Tweets and get Compound, Positive, Negative, & Neutral Scores
def sentiment_analyzer(df):
    # Setup sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Variables for holding sentiments
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []

    # Loop through Tweets
    for text in df['Text']:

    # Run Vader Analysis on each tweet
    results = analyzer.polarity_scores(text)
    compound = results["compound"]
    pos = results["pos"]
    neu = results["neu"]
    neg = results["neg"]

    # Add each value to the appropriate list
    compound_list.append(compound)
    positive_list.append(pos)
    negative_list.append(neg)
    neutral_list.append(neu)

    df['Compound Score'] = compound_list
    df['Positve Score'] = positive_list
    df['Negative Score'] = negative_list
    df['Neutral Score'] = neutral_list

    return df

def user_tweets(user, api, consumer_key, consumer_secret, access_token, access_token_secret,endpage = 26):
    target_user = ('@'+user)

    # List to store dictionaries of tweets
    tweets = []

    # Loop through 25 pages of tweets and grab 500 tweets
    for x in range(1, endpage):

    for tweet in api.user_timeline(target_user, page=x, tweet_mode='extended'):
      tweets.append(tweet)

    # Convert list of dicitonary tweets into a dataframe
    tweet_df = df_creator(tweets)

    # Reset index to date created for Group By purposes
    tweet_df = tweet_df.set_index('Created')

    tweet_df = sentiment_analyzer(tweet_df)

    final_df = tweet_df.resample('Y').mean()

    final_df['Handle'] = user

    final_df.index = [date.year for date in final_df.index]

    return final_df[['Handle','Likes', 'Retweet', 'Compound Score', 'Positve Score', 'Negative Score','Neutral Score']]

# Convert Pandas DF to Png format
def render_mpl_table(data,col_width=3.0,row_height=0.625,font_size=14,header_color='#40466e',
row_colors=['#f1f1f2', 'w'],edge_color='w',bbox=[0, 0, 1, 1],header_columns=0,ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

##########################################################
# Base Bot
##########################################################

def post_analysis(username):

    response = user_tweets(user=username,api=api,consumer_key=consumer_key,consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret, endpage = 200)

    # Render response to a time series chart
    ax = response.plot()
    fig = ax.get_figure()
    fig.savefig('time_plot.png')

    # Render response to PNG
    render_mpl_table(response).get_figure().savefig('table.png')

    # Post DataFrame
    api.update_with_media('table.png', f'Sentimental Analysis of @{username}')
    api.update_with_media('time_plot.png', f'Time Series Graph of @{username}')


def find_completed_requests():
    tweets = api.user_timeline(rpp=1000)

    completed_requests = set()
    for tweet in tweets:
        if 'labels for' not in tweet['text']:
            continue
        for user_mention in tweet['entities']['user_mentions']:
            if user_mention['screen_name'] != me:
                completed_requests.add(user_mention['screen_name'])

    return completed_requests


def find_next_request():
    tweets = api.search(f'@{me} Analyze:')['statuses']

    requests = set()
    for tweet in tweets:
        for user_mention in tweet['entities']['user_mentions']:
            if user_mention['screen_name'] != me:
                requests.add(user_mention['screen_name'])

    new_requests = requests - find_completed_requests()

    try:
        return new_requests.pop()
    except:
        return None


while True:
    print("Updating Twitter")

    next_request = find_next_request()
    print('Next Request:', next_request)

    if next_request:
        post_analysis(next_request)

    time.sleep(20)
