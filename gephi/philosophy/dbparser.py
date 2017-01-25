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

influencedfilein = "./philosophersdirty.txt"
influencedfileout = "./philosophers.csv"

results = readfile(influencedfilein)
languageoutput = writefile(influencedfileout)
sys.exit()
        
