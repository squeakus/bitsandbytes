# Regular Expression parser that tags the generated tokens
# borrowed from: 
#http://stackoverflow.com/questions/526469/practical-examples-of-nltk-use

import nltk
import pprint

tokenizer = None
tagger = None

def init_nltk():
    global tokenizer
    global tagger
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')
    tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())

def tag(text):
    global tokenizer
    global tagger
    if not tokenizer:
        init_nltk()

    tokenized = tokenizer.tokenize(text)    
    tagged = tagger.tag(tokenized)
    return tagged

def main():
    sample_file = open('sample_text_2.txt', 'r')
    text = sample_file.read()
    tagged = tag(text)
    for word in tagged:
        print word[1]

    #if you want the list sorted
    #l = list(set(tagged))
    #l.sort(lambda x,y:cmp(x[1],y[1]))
    #pprint.pprint(tagged)

if __name__ == '__main__':
    main()
