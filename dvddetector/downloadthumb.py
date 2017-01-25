import urllib2, json

dvdname = "KISS_KISS_BANG_BANG"
searchquery = "https://www.googleapis.com/customsearch/v1?key=AIzaSyBUts7RpJt7CvITxv1WP0NAlgMQJ6TjLwA&cx=004163519135800887416:2b9zuir4iuy&alt=json&q="+dvdname

print searchquery

googleresult = urllib2.urlopen(searchquery)
resultdict = json.load(googleresult)

for result in resultdict:
    print result
