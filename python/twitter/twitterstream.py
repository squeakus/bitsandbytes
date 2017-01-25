#!/usr/bin/python

#-----------------------------------------------------------------------
# twitter-stream-format:
#  - ultra-real-time stream of twitter's public timeline.
#    does some fancy output formatting.
#-----------------------------------------------------------------------

from twitter import *
import re
from requests.auth import OAuth1


search_term = "goldsmiths"

# import a load of external features, for text display and date handling
from time import strftime
from textwrap import fill
from termcolor import colored
from email.utils import parsedate

# these tokens are necessary for user authentication
consumer_key = u'y8KlfLYrBdjCKAMIWPt91Q'
consumer_secret = u'ApV8s1Hg4WsR7jAY0vKXmePiGc8BuCBMbrkky55iehQ'
access_key = u'182309114-zBJTzx72PGS3p5yTikSMi6lTXPEIR3f92Ky8KsLU'
access_secret = u'2R9TwGSfvH7z8eDpMaHsoHFwRLA2r7bMixG4wfIhJU'

# create twitter API object
auth = OAuth1(consumer_key, consumer_secret,access_key, access_secret, signature_type='query')
stream = TwitterStream(auth = auth, secure = True)

# iterate over tweets matching this filter text
# IMPORTANT! this is not quite the same as a standard twitter search
tweet_iter = stream.statuses.filter(track = search_term)

pattern = re.compile("%s" % search_term, re.IGNORECASE)

for tweet in tweet_iter:
	# check whether this is a valid tweet
	if tweet.get('text'):

		# turn the date string into a date object that python can handle
		timestamp = parsedate(tweet["created_at"])
		# now format this nicely into HH:MM:SS format
		timetext = strftime("%H:%M:%S", timestamp)

		# colour our tweet's time, user and text
		time_colored = colored(timetext, color = "white", attrs = [ "bold" ])
		user_colored = colored(tweet["user"]["screen_name"], "green")
		text_colored = tweet["text"]
		# replace each instance of our search terms with a highlighted version
		text_colored = pattern.sub(colored(search_term.upper(), "yellow"), text_colored)

		# add some indenting to each line and wrap the text nicely
		indent = " " * 11
		text_colored = fill(text_colored, 80, initial_indent = indent, subsequent_indent = indent)

		# now output our tweet
		print "(%s) @%s" % (time_colored, user_colored)
		print "%s" % (text_colored)
