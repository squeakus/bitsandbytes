import operator
bible = open('bible.txt','r')
bibdict = {}

for line in bible:
    line = line.split(' ')

    for word in line:
        word = word.rstrip()
        word = word.rstrip('\r\n')
    	if word in bibdict:
	   bibdict[word] += 1
	else:
	   bibdict[word] = 1

sorted_x = sorted(bibdict.items(), key=operator.itemgetter(1))
for elem in sorted_x:
    print elem
