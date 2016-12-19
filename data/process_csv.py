import re

file1 = '94043399802_facebook_statuses.csv'
file2 = '94043399802_facebook_comments.csv'

files = [file1, file2]

for file in files:

	with open(file) as f:
		lines = f.readlines()
	
	with open('processed_' + file)
		for line in lines:
			newline = line.lower()
			newline = re.sub(r'/\w+\s*\w*"', "", newline)
			newline = re.sub(r'\W', "", newline)
