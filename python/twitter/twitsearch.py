import urllib.request, urllib.error, urllib.parse, json, pprint
HASHTAG = "garda"
u = urllib.request.urlopen('http://search.twitter.com/search.json?q='+HASHTAG+'&rpp=25')
resultdict = json.load(u)

pprint.pprint(resultdict)
for tweet in resultdict['results']:
    print(tweet['text'])
