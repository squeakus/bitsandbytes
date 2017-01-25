import nltk.text
from nltk.corpus import gutenberg

print gutenberg
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
print moby
print moby.concordance('whale')
print moby.findall("<.*><.*><whale>")

#for token in moby:
#    print token

