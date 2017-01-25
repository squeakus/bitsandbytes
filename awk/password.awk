#awk '{ print }' /etc/passwd
#awk '{ print $0 }' /etc/passwd
#o/p completely unrelated to input
awk '{ print "hiya" }' /etc/passwd 
