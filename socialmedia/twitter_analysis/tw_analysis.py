# -*- coding: utf-8 -*-

import pandas as pd
import string
import operator

# After downloading out tweets from twitter into a csv. I have no data so I can't do this


def load_tweets(tweet_file):
    
    """ Load and process a Twitter analytics data file """
    
    # Read tweet data (obtained from Twitter analytics)
    tweet_df = pd.read_csv(tweet_file)
    
    # Drop irrelevant columns
    tweet_df = tweet_df.drop(tweet_df.columns[[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38, 39]], axis=1)
    
    return tweet_df
    

def get_total_tweets(df):
    """Returns and prints the total tweets the total tweets"""
    total = len(df.index)
    # print('Total tweets this period: ', total, '\n')
    return total
    

def get_retweets(df=None, top=5):
    """
    Mean retweets per tweet and top (default 5) retweeted tweets.
    :param df: The DataFrame to use
    :param top: The count of retweeted tweets. By default 5
    :return: The mean and a df with top tweets. Default 5
    """

    tdf = df.sort_values(by='retweets', ascending=False)
    tdf = tdf.reset_index(drop=True)
    mean = round(tdf['retweets'].mean(),2)

    top5 = tdf.head(5)
    return mean, top5
    

def print_retweets(top_df=None):
    """
    Prints the top df rows
    :param top_df: The DataFrame with top tweets
    :return: None
    """
    for i in range(len(top_df.index)):
        print(top_df['Tweet text'].ix[i], '-', top_df['retweets'].ix[i])
    print('\n')
    return None


def get_likes(df=None, top=5):
    """
    Mean likes per tweet and top (default 5) likes.
    :param df: The DataFrame to use
    :param top: The count of top likes. By default 5
    :return: The mean and a df with top likes. Default 5
    """
    tdf = df.sort_values(by='likes', ascending=False)
    tdf = tdf.reset_index(drop=True)
    mean = round(tdf['likes'].mean(),2)
    top5 = tdf.head(5)
    return mean, top5


def print_likes(top_df=None):
    """
    Prints the top df rows
    :param top_df: The DataFrame with top tweets
    :return: None
    """
    for i in range(len(top_df.index)):
        print(top_df['Tweet text'].ix[i], '-', top_df['likes'].ix[i])
    print('\n')
    return None


def get_impressions(df=None, top=5):
    """
    Mean impressions per tweet and top (default 5) likes.
    :param df: The DataFrame to use
    :param top: The count of top impressions. By default 5
    :return: The mean and a df with top impressions. Default 5
    """
    tdf = df.sort_values(by='impressions', ascending=False)
    tdf = tdf.reset_index(drop=True)
    mean = round(tdf['impressions'].mean(),2)
    top5 = tdf.head(top)
    return mean, top5


def print_impressions(top_df=None):
    """
    Prints the top df rows
    :param top_df: The DataFrame with top impressions
    :return: None
    """
    for i in range(len(top_df.index)):
        print(top_df['Tweet text'].ix[i], '-', top_df['impressions'].ix[i])
    print('\n')
    return None


def get_tags(df=None):
    """
    Return a dictionary with the hashtags
    :param df: The DataFrame
    :return: The tags dictionary
    """
    if df is None:
        return None
    tag_dict = {}
    for i in df.index:
        tweet_text = df.ix[i]['Tweet text']
        tweet = tweet_text.lower()
        tweet_tokenized = tweet.split()

        for word in tweet_tokenized:
            if word[0:1] == '#' and len(word)>1:
                key = word.translate(string.maketrans("", ""), string.punctuation)
                if key in tag_dict:
                    tag_dict[key] += 1
                else:
                    tag_dict[key] = 1
    return tag_dict

def get_mentions(df=None):
    """
    Return a dictionary with the hashtags
    :param df: The DataFrame
    :return: The mentions dictionary
    """
    if df is None:
        return None
    mentions_dict = {}
    for i in df.index:
        tweet_text = df.ix[i]['Tweet text']
        tweet = tweet_text.lower()
        tweet_tokenized = tweet.split()

        for word in tweet_tokenized:
            if word[0:1] == '@' and len(word)>1:
                key = word.translate(string.maketrans("", ""), string.punctuation)
                if key in mentions_dict:
                    mentions_dict[key] += 1
                else:
                    mentions_dict[key] = 1
    return mentions_dict


def print_tags(tag_dict, limit=10):
    top_tags = dict(sorted(tag_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:limit])
    top_tags_sorted = sorted(top_tags.items(), key=lambda x: x[1])[::-1]
    print('Top Tags:')
    print('---------')
    for tag in top_tags_sorted:
        print(tag[0], '-', str(tag[1]))


def print_mentions(mentions_dict, limit=10):
    top_mentions = dict(sorted(mentions_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:limit])
    top_mentions_sorted = sorted(top_mentions.items(), key=lambda x: x[1])[::-1]
    print('Top mentions:')
    print('---------')
    for mention in top_mentions_sorted:
        print(mention[0], '-', str(mention[1]))


def time_analysis(df):
    gmt_offset = 2

    # Create datetime column
    df['ts'] = pd.to_datetime(df['time'])
    df['ts'] = df.ts + pd.to_timedelta(gmt_offset, unit='h')

    # Add hour of day and day of week columns
    df['hod'] = [t.hour for t in df.ts]
    df['dow'] = [t.dayofweek for t in df.ts]

    hod_dict = {}
    hod_count = {}
    dow_dict = {}
    dow_count = {}
    weekday_dict = {0:'Δευ', 1:'Τρι', 2:'Τετ', 3:'Πεμ', 4:'Παρ', 5:'Σαβ', 6:'Κυρ'}

    # Process tweets, collect stats
    for i in df.index:
        hod = df.ix[i]['hod']
        dow = df.ix[i]['dow']
        imp = df.ix[i]['impressions']

        if hod in hod_dict:
            hod_dict[hod] += int(imp)
            hod_count[hod] += 1
        else:
            hod_dict[hod] = int(imp)
            hod_count[hod] = 1

        if dow in dow_dict:
            dow_dict[dow] += int(imp)
            dow_count[dow] += 1
        else:
            dow_dict[dow] = int(imp)
            dow_count[dow] = 1

    print('Average impressions per tweet by hour tweeted:')
    print('----------------------------------------------')
    for hod in hod_dict:
        print(hod, '-', hod+1, ':', hod_dict[hod]/hod_count[hod], '=>', hod_count[hod], ' tweets')

    print('\nAverage impressions per tweet by day of week tweeted:')
    print('-----------------------------------------------------')
    for dow in dow_dict:
        print(weekday_dict[dow], ':', dow_dict[dow]/dow_count[dow], '=>', dow_count[dow], ' tweets')


def main():
    tweet_df = load_tweets('tweet_20170601_20170701_en.csv')
    # print(tweet_df.head())
    print(tweet_df.columns)

if __name__ == '__main__':
    main()