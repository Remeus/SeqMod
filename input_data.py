"""Preprocess the data sets"""
import os
from collections import Counter
import pickle
import re

def read_data(fname, count, word2idx):
    """Read and preprocess the datasets"""
    topk = 10000 # Max vocabulary size

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            loaded_data = pickle.load(f)
    else:
        raise("[!] Data %s not found" % fname)

    data = []
    words = []

    n_total = 0
    for ind, row in enumerate(loaded_data):

        data.append([])

        try:
            line_status = row[0]
            line_comment = row[1]
        except: # No status or comment
            continue

        if ind % (len(loaded_data) // 10) == 0:
            print('%d %% done...' % ((100 * ind) / len(loaded_data)))

        lines = [line_status, line_comment]

        for line in lines:
            data[ind].append([])
            words.extend(line.split())

        n_total += 1

    if len(count) == 0:
        count.append(['<eos>', 0])
        count.append(['<unk>', 0])

    count[0][1] += len(lines) * n_total
    count.extend(Counter(words).most_common(topk))

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    if len(set(words)) > topk:
        word2idx['<unk>'] = 1
        n_unk = 0

    for ind, row in enumerate(loaded_data):

        try:
            line_status = row[0]
            line_comment = row[1]
        except: # No comment
            continue

        lines = [line_status, line_comment]

        for i, line in enumerate(lines):
            for word in line.split():
                if word not in word2idx:
                    word = '<unk>'
                    n_unk += 1
                index = word2idx[word]
                data[ind][i].append(index)
            data[ind][i].append(word2idx['<eos>'])

    # Remove empty lists (errors)
    data = [x for x in data if x != []]

    # Add <unk> to count
    count[1][1] += n_unk

    print("Read %s words from %s" % (sum(sum(len(d) for d in d2) for d2 in data), fname))

    with open('preloaded_' + fname, 'wb') as f:
        pickle.dump(data, f)
        pickle.dump(word2idx, f)
        pickle.dump(count, f)

    return data



def convert_question(question, word2idx):
    """Convert a question for prediction"""
    # Process question
    question = question.lower()
    question = re.sub(r'/\w+\s*\w*$', "", question)  # Signature
    question = re.sub(r'\n', "", question)  # Newline
    question = re.sub(r'[^\w\s\d]', " ", question)  # Neither letter nor whitespace
    question = re.sub(r'\d+(\s\d+)*', "<unk>", question)  # Numbers
    # Convert question
    res = []
    for word in question.split():
        if word not in word2idx:
            word = '<unk>'
        index = word2idx[word]
        res.append(index)
    res.append(word2idx['<eos>'])
    return res