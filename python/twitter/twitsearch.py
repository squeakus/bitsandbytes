import urllib.request, urllib.error, urllib.parse, json, pprint
HASHTAG = "garda"
query = 'https://api.twitter.com/1.1/search/tweets.json?q='+HASHTAG+'&rpp=25'
print("search:", query)
u = urllib.request.urlopen(query)
resultdict = json.load(u)

pprint.pprint(resultdict)
for tweet in resultdict['results']:
    print(tweet['text'])
