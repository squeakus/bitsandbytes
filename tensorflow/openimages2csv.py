

def main():


	with open('sub-test-annotations-bbox.csv', 'r') as infile:

		headers = infile.readline()
		headers = headers.rstrip().split(',')
		print(headers)

		for line in infile:
			line = line.split(',')
			filename = line[0] + '.jpg'
			label = line[-1].rstrip()
			# print(filename, label)



if __name__ == '__main__':
    main()
