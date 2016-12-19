import pandas as pd
import numpy as np
import pickle
from random import shuffle
import re


def process(txt):
	res = txt.lower()
	res = re.sub(r'/\w+\s*\w*$', "", res) # Signature
	res = re.sub(r'\n', "", res) # Newline
	res = re.sub(r'[^\w\s\d]', " ", res) # Neither letter nor whitespace
	res = re.sub(r'\d+(\s\d+)*', "<unk>", res)  # Numbers
	return res


statuses = pd.read_csv('94043399802_facebook_statuses.csv', sep=',', header=0, quotechar='"')
comments = pd.read_csv('94043399802_facebook_comments.csv', sep=',', header=0, quotechar='"')

print('%d statuses loaded' % len(statuses))
print('%d comments loaded' % len(comments))

statuses_customer = statuses[statuses['status_author'] != 'Telenor Norge']
statuses_customer_answered = statuses_customer[statuses_customer['num_comments'] > 0]

print('%d statuses of customer have been answered' % len(statuses_customer_answered))

comments_telenor = comments[comments['comment_author'] == 'Telenor Norge']

# comments_telenor_sorted = comments_telenor.sort_values('comment_published')

comments_telenor_first = comments_telenor[comments_telenor['parent_id'].isnull()]

print('%d comments of Telenor for the statuses' % len(comments_telenor_first))

res = []
for i, s in statuses_customer_answered.iterrows():
	c = comments_telenor_first[comments_telenor_first['status_id'] == s['status_id']]
	if len(c) == 0:
		continue
	s_str = s['status_message']
	if type(s_str) is not str:
		continue
	s_str = process(s_str)
	c_str = process(c['comment_message'].values[0])
	res.append([s_str, c_str])
	if i % 1000 == 0:
		print('%d %% done' % ((100 * i) // len(statuses_customer_answered)))
		print(s_str)
		print(c_str)
	
shuffle(res)

size_training = int(0.9 * len(res))	
print('Training size: %d' % size_training)
with open('train.pickle', 'wb') as f:
	pickle.dump(res[:size_training], f)
with open('val.pickle', 'wb') as f:
	pickle.dump(res[size_training:], f)
