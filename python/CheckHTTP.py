from httplib import HTTP 
from urlparse import urlparse 

def checkURL(url): 
     p = urlparse(url)
     h = HTTP(p[1])
     h.putrequest('HEAD', p[2]) 
     h.endheaders()
     reply =  h.getreply()
     print "reply:",  reply
     print url, "response code:", reply[0]

if __name__ == '__main__': 
     checkURL('http://slashdot.org') 
     checkURL('http://www.reddit.com') 
     checkURL('http://www.google.com')
     checkURL('http://slashdot.org/notadirectory') 
     checkURL('http://www.iuh3r498h438igybi3b3km.com') 
## end of http://code.activestate.com/recipes/101276/ }}}


