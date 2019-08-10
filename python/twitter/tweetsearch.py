import tweepy
import csv
import pandas as pd
from config import *

def main():
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)

	public_tweets = api.home_timeline()
	for tweet in public_tweets:
	    print(tweet.text)

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth,wait_on_rate_limit=True)
	#####United Airlines
	# Open/Create a file to append data
	csvFile = open('garda.csv', 'a')
	#Use csv Writer
	csvWriter = csv.writer(csvFile)

	for tweet in tweepy.Cursor(api.search,q="garda",count=100,
	                           lang="en",
	                           since="2017-04-03",
	                           tweet_mode='extended',
	                           full_text=True).items():
		#.encode('ascii', 'ignore')
		filtered = filter_tweet(tweet.full_text)
		print (tweet.created_at, filtered)
		csvWriter.writerow([filtered])

def filter_tweet(text):
	#text.replace('\\xe2\\x80\\xa6', '')
	
	result = []
	text = text.split()
	print ("TEXT", text)
	for elem in text:
		append = True
		if elem == 'RT':
			append = False
		if elem.startswith('@'):
			append = False
		if elem.startswith('\\x'):
			print("found one", elem)
			append = False
		if '\\x'  in elem:
			print("found one", elem)
			append = False
		if elem.startswith('http'):
			append = False
		if append:
			result.append(elem)

	result_string = ' '.join(result)
	return result_string
if __name__ == '__main__':
	main()