from nltk.tokenize import *
from nltk import word_tokenize, wordpunct_tokenize

s = ("Good muffins cost $3.88\nin New York.  Please buy me\n"
     "two of them.\n\nThanks.")

#watch out for UTF8
utf_string = "das ist ein t\xc3\xa4ller satz"
print "unicode:", wordpunct_tokenize(utf_string.decode('utf8'))
print "utf8:   ", wordpunct_tokenize(utf_string)


#standard tokenizers

print "word       :", word_tokenize(s) 
# regexp: '\w+|[^\w\s]+'
print "word_punct :", wordpunct_tokenize(s)
print "whitespace :", WhitespaceTokenizer().tokenize(s)
print "space      :", SpaceTokenizer().tokenize(s)
print "treebank   :", TreebankWordTokenizer().tokenize(s)
print "blank lines:", LineTokenizer(blanklines='keep').tokenize(s)
print "lines      :", LineTokenizer(blanklines='discard').tokenize(s)
print "tabs       :", TabTokenizer().tokenize('a\tb c\n\t d')

print "\nregexp tokenizers"
capword_tokenizer = RegexpTokenizer('[A-Z]\w+')
print "Capwords:", capword_tokenizer.tokenize(s)

#alphabetic sequences, money expressions, and non-whitespace
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
print "all     :", tokenizer.tokenize(s)  

tokenizer = RegexpTokenizer('\s+', gaps=True)
print "no gaps :", tokenizer.tokenize(s)  
