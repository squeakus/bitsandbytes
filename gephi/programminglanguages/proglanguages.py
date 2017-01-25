import re,sys
from collections import Counter

"""
Author: Brendan Griffen
Date: 2014/01/19 1:55 AM EST
Copyright (c) 2014 Brendan Griffen. All rights reserved.

Description: Parses over data pulled from dpbedia queries to create csv files which can be loaded into Gephi.
One file is for connecting programming languages via "influence" and "influenced by" terms on dbpedia.

Usage: python proglanguages.py

Notes:
Directories will have to be constructed manually.
"""

def readfile(filename):
    results = []
    with open(filename) as inputfile:
        for line in inputfile:
            results.append(line[1:].strip().split(':'))
    
    return results

def writefile(fileout):
    reptxtlist=["[http]","(programming language)"]
    f = open(fileout,'w')
    languageoutput = []
    #iterate through results
    for result in results:
        #iterate through replacement text list
        for reptxt in reptxtlist:
            #iterate over each result in line [2 cases]
            for i,res in enumerate(result):
                if reptxt in res:
                    #inner to outer: remove "_", strip string, remove replacement text if any, remove things in ()
                    res = re.sub(' +',' ',re.sub(r'\([^)]*\)', '',res.replace(reptxt,"").strip().replace("_"," ")))
    
                result[i] = res
    
        #only write different entries out as the clean could reveal self-connections
        if result[0] != result[1]:
            writeline = '"'+result[0].strip() +'"' + ',' + '"'+result[1].strip()+'"'
            languageoutput.append(result[0].strip()+','+result[1].strip())
            f.write(writeline + "\n")
    
    f.close()
    return languageoutput

influencedfilein = "./data/languages/influencedURL.txt"
influencedfileout = "./data/languages/programminglanguagesURL.csv"

results = readfile(influencedfilein)
languageoutput = writefile(influencedfileout)
sys.exit()
designedfilein = "./data/desingers/designedby.txt"
designedfileout = "./data/desingers/designerinfluencers.csv"
results = readfile(designedfilein)

# list of characters we want removed from raw data
reptxtlist=["[http]"]
output= []
for result in results:
    #iterate through replacement text list
    for reptxt in reptxtlist:
        #iterate over each result in line [2 cases]
        for i,res in enumerate(result):
            if reptxt in res:
                #inner to outer: remove "_", strip string, remove replacement text if any, remove things in ()
                res = re.sub(' +',' ',re.sub(r'\([^)]*\)', '',res.replace(reptxt,"").strip().replace("_"," ")))
            #update result
            result[i] = res
         
    #only write different entries out as the clean could reveal self-connections
    if result[0] != result[1]:
        writeline = '"'+result[0].strip() +'"' + ',' + '"'+result[1].strip()+'"'
        output.append(result[0].strip()+','+result[1].strip())

personlist = []
languagelist = []
for i in xrange(0,len(output)):
    personlist.append(output[i].split(',')[1])
    languagelist.append(output[i].split(',')[0])

# Construct unique lists of people and languages
personlistunique = set(personlist)
languagelistunique = set(languagelist)

f = open(designedfileout,'w')

#iterate of each person
for person in personlistunique:
    influencedlang = []
    #iterate over language-people lists
    for i in xrange(0,len(output)):
        if person == output[i].split(',')[1]:
           influencedlang.append(output[i].split(',')[0])

    #construct list of people that each person is connected to
    influencedpeople = []
    for sublang in influencedlang:
        for i in xrange(0,len(output)):
            if sublang == output[i].split(',')[0]:
                influencedpeople.append(output[i].split(',')[1])

    #make sure the person is not self-referencing
    strwrite = ''
    for item in influencedpeople:
        if item != person:
            strwrite = strwrite + '"' +item +'"'  ','

    if len(strwrite)>0:
        print person + "," + strwrite
        f.write('"' +person +'"'  ',' + strwrite)
        f.write('\n')

f.close()
        
