import os
from tqdm import tqdm
import json
import re

def my_tokenize(txt):
    return re.split(r'\s+', txt)

def text_tokenize_and_postagging(txt, sent_start):
    new_txt = txt.strip()
    new_txt = re.sub(r"\'|\"", ',', new_txt)

    tokens=my_tokenize(new_txt)

    return tokens

def token_from_sent(txt, sent_start):

    return text_tokenize_and_postagging(txt, sent_start)

def preprocess(fileName, out_name):
    fp = open(fileName, 'r')

    fp_out = open(out_name, 'w')


    for line in tqdm(fp.readlines()):

        json_line = json.loads(line)

        articleId = json_line['articleId'][json_line['articleId'].rfind('/')+1:]
        sentId = json_line['sentId']
        sentText = json_line['sentText']
        tmp_tokens = token_from_sent(sentText, 0)

        for token in tmp_tokens:
            if token.isalpha():
                fp_out.write(token+" ")




    fp.close()
    fp_out.close()


def preprocess_all(fileName, testfilename, out_name, alpha):
    fp = open(fileName, 'r')

    fp_out = open(out_name, 'w')

    for line in tqdm(fp.readlines()):

        json_line = json.loads(line)

        articleId = json_line['articleId'][json_line['articleId'].rfind('/') + 1:]
        sentId = json_line['sentId']
        sentText = json_line['sentText']
        tmp_tokens = token_from_sent(sentText, 0)

        for token in tmp_tokens:
            if alpha == False:
                fp_out.write(token + " ")
            else:
                if token.isalpha():
                    fp_out.write(token + " ")

    fp.close()


    fp = open(testfilename, 'r')

    for line in tqdm(fp.readlines()):

        json_line = json.loads(line)

        articleId = json_line['articleId'][json_line['articleId'].rfind('/') + 1:]
        sentId = json_line['sentId']
        sentText = json_line['sentText']
        tmp_tokens = token_from_sent(sentText, 0)

        for token in tmp_tokens:
            if alpha == False:
                fp_out.write(token + " ")
            else:
                if token.isalpha():
                    fp_out.write(token + " ")

    fp.close()


    fp_out.close()




# preprocess("/Users/feili/dataset/cotype/NYT/train.json", "/Users/feili/project/word2vec/nyt_train.txt")

preprocess_all("/Users/feili/dataset/cotype/NYT/train.json", "/Users/feili/dataset/cotype/NYT/test.json", "/Users/feili/project/word2vec/nyt_all.txt", False)

preprocess_all("/Users/feili/dataset/cotype/NYT/train.json", "/Users/feili/dataset/cotype/NYT/test.json", "/Users/feili/project/word2vec/nyt_onlyalpha.txt", True)