import nltk.tag, nltk.data
from nltk.corpus import brown


brown_news_tagged = brown.tagged_sents(categories='news')
brown_news_text = brown.sents(categories='news')

print "training sets:",brown_news_text
print "tagged:", brown_news_tagged


brown_train = brown_news_tagged[100:]
brown_test = brown_news_tagged[:100]

regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
     (r'(The|the|A|a|An|an)$', 'AT'),   # articles
     (r'.*able$', 'JJ'),                # adjectives
     (r'.*ness$', 'NN'),                # nouns formed from adjectives
     (r'.*ly$', 'RB'),                  # adverbs
     (r'.*s$', 'NNS'),                  # plural nouns
     (r'.*ing$', 'VBG'),                # gerunds
     (r'.*ed$', 'VBD'),                 # past tense verbs
     (r'.*', 'NN')                      # nouns (default)
     ])

tagger = nltk.tag.UnigramTagger(brown_train, backoff=regexp_tagger)
print(tagger.tag(['select', 'the', 'files']))

