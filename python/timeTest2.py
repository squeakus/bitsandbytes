import datetime

now = datetime.datetime.now()
print "now:",now
print repr(now)
print type(now)
print now.year, now.month, now.day
print now.hour, now.minute, now.second
print now.microsecond
filename= "run_"+str(now.day)+"_"+str(now.month)+"_"+str(now.hour)+str(now.minute)+str(now.second)
print filename
