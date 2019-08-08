
import requests, json, sys, re, enchant, langid
from requests.auth import OAuth1

# import a load of external features, for text display and date handling
from time import strftime
from textwrap import fill
from termcolor import colored
from email.utils import parsedate

#set it up for unicode
reload(sys)
sys.setdefaultencoding("utf-8")

queryoauth = OAuth1(consumer_key, consumer_secret,
                    access_key, access_secret,
                    signature_type='query')

r = requests.post('https://stream.twitter.com/1/statuses/sample.json',
                  auth=queryoauth, prefetch=False)

def highlight_word(tweet, search_term, pattern):
    # turn the date string into a date object that python can handle
    timestamp = parsedate(tweet["created_at"])
    # now format this nicely into HH:MM:SS format
    timetext = strftime("%H:%M:%S", timestamp)
    # colour our tweet's time, user and text
    time_colored = colored(timetext, color = "white", attrs = [ "bold" ])
    user_colored = colored(tweet["user"]["screen_name"], "green")
    text_colored = tweet["text"]
    # replace each instance of our search terms with a highlighted version
    text_colored = pattern.sub(colored(search_term.upper(), "red"), 
                               text_colored)
   
    # now output our tweet
    print("(%s) @%s %s" % (time_colored, user_colored, text_colored))

def get_location(tweet):
    username = tweet["user"]["screen_name"]
    user_query = 'https://api.twitter.com/1/users/show.json?id=' + username

    result = requests.get(user_query,auth=queryoauth)
    user_info = result.json
    print(user_info['name'], user_info['location'])

def get_user_tweets(tweet, count):
    username = tweet["user"]["screen_name"]
    tweet_query = 'https://api.twitter.com/1/statuses/user_timeline.json?include_entities=true&include_rts=false&screen_name='+username+'&count='+'count'    
    result = requests.get(tweet_query,auth=queryoauth)
    tweet_list = result.json
    for tweet in tweet_list:
        print(tweet['text'])
    return tweet_list
    
def quick_print(tweet):
    #if int(tweet.get('retweet_count')) == 0:
    #if not tweet['coordinates'] == None:
    langtest = langid.classify(tweet['text'])
    language = langtest[0]
    if language == 'en':
        print(colored(tweet['text'],"green"))
    else:
        print(colored(tweet['text'],"red"))
        #get_user_tweets(tweet, 5)
    get_location(tweet)

    
def dictionary_check(tweet, d):
    print(tweet['text'])
    word_list = tweet['text'].split(' ')
    total_words = len(word_list)
    #english_words = sum([d.check(word) for word in word_list]
    print(total_words)
        
search_term = "and"
#pattern = re.compile(search_term, re.IGNORECASE)
d = enchant.Dict("en_US")
pattern = re.compile(search_term)


for line in r.iter_lines():
    if line: # filter out keep-alive new lines
        tweet = json.loads(line)
        if  tweet.get('text'):
            if tweet['user']['lang'] == 'en':
                if not tweet['text'][0:2] == 'RT':
                    #print  tweet['user']['lang']
                    quick_print(tweet)
                    #dictionary_check(tweet, d)
                    #highlight_word(tweet, search_term, pattern)

            
            # for idx, feature in enumerate(tweet):
            #     print idx, ":", feature, str(tweet.get(feature))
