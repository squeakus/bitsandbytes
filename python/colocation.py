word_list = []
word_count = {}
pair_count = {}
special_chars = ['.', ',', '\'', '-']
textfile = open('Robotics_article_1.txt','r')

for line in textfile:
    #sanitize input (get rid of full stops and new lines)
    line = line.rstrip('\r\n')
    for special in special_chars:
        line.replace(special,'')

    #break up line (tokenise it)
    line = line.split(' ')
    for word in line:
        word = word.lower()
        word_list.append(word)


        #now count the words!
for idx, word in enumerate(word_list):
    # if word already in dictionary then increment
    if word in word_count:
        word_count[word] += 1
    # otherwise add it to the dictionary
    else:
        word_count[word] = 1

    # look at next word but dont go off the end of the list
    if idx + 1 < len(word_list):
        nextword = word_list[idx+1]

    # if word pair already on the dictionary then increment
    if (word, nextword) in pair_count:
        pair_count[word, nextword] +=1
    #otherwise add it to the dictionary
    else:
        pair_count[word, nextword] = 1
        
sorted_words = sorted(word_count.items(), key=lambda item: item[1])
sorted_pairs = sorted(pair_count.items(), key=lambda item: item[1])

for word in sorted_pairs:
    print word
