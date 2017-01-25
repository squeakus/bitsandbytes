#!/usr/bin/env python
import urllib

def checkUrl(url):
    response_code = urllib.urlopen(url).getcode()
    print url, "response code:", response_code

if __name__=="__main__":
    checkUrl("http://www.google.com") # True
    checkUrl("http://www.reddit.com") # True
    checkUrl('http://ncra.ucd.ie/members/byrnej.html')
    checkUrl('http://ncra.ucd.ie/members/byrnejj.html')
    #checkUrl('ncra.ucd.ie/members/byrnejj.html') IT NEEDS THE HTTP!
    checkUrl("http://www.google.com/ilikecheese.html") 
    checkUrl("http://www.reddit.com/monvf")
