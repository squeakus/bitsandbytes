import urllib2, json, pprint
HASHTAG = "garda"
u = urllib2.urlopen('http://search.twitter.com/search.json?q='+HASHTAG+'&rpp=25')
resultdict = json.load(u)

pprint.pprint(resultdict)
for tweet in resultdict['results']:
    print tweet['text']
