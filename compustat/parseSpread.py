def tidy_line(raw_line):
    raw_line = raw_line.rstrip()
    raw_line = raw_line.replace(",,",",")
    return raw_line.split(',')

def parse_csv(csv_file):
    csv_file = open(csv_file,'r')
    #ignore the first line
    csv_file.readline()
    #get the column headers from the file
    header = tidy_line(csv_file.readline())
    #An array of dictionaries stores the results
    result = []

    for line in csv_file:
        dictionary = {}
        row = tidy_line(line)
        for i in range(len(header)):
            dictionary[header[i]] = row[i]
        result.append(dictionary)
    return result

RESULTS = parse_csv("test2.csv")

for entry in RESULTS:
    print entry['Company Name']
