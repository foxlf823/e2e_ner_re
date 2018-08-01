#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import nltk
import my_utils
import pandas as pd
import logging
import os
import shutil
from tqdm import tqdm
import my_utils1
from utils.functions import normalize_word
import re

# set([u'PERSON', u'LOCATION', u'ORGANIZATION'])
# set([u'/business/company/founders', u'/people/person/place_of_birth', u'/people/deceased_person/place_of_death', u'/business/company_shareholder/major_shareholder_of', u'/people/ethnicity/people', u'/location/neighborhood/neighborhood_of', u'/sports/sports_team/location', u'/business/person/company', u'/business/company/industry', u'/business/company/place_founded', u'/location/administrative_division/country', u'None', u'/sports/sports_team_location/teams', u'/people/person/nationality', u'/people/person/religion', u'/business/company/advisors', u'/people/person/ethnicity', u'/people/ethnicity/geographic_distribution', u'/people/person/place_lived', u'/business/company/major_shareholders', u'/people/person/profession', u'/location/country/capital', u'/location/location/contains', u'/location/country/administrative_divisions', u'/people/person/children'])
def statDataset(trainFile, testFile):
    entityType = set()
    relationType = set()

    fp = open(trainFile, 'r')

    for line in tqdm(fp.readlines()):
        json_line = json.loads(line)

        for json_entity in json_line['entityMentions']:

            entity_type = json_entity['label']
            entityType.add(entity_type)


        for json_relation in json_line['relationMentions']:

            label = json_relation['label']
            relationType.add(label)

    fp.close()


    fp = open(testFile, 'r')

    for line in tqdm(fp.readlines()):
        json_line = json.loads(line)

        for json_entity in json_line['entityMentions']:

            entity_type = json_entity['label']
            entityType.add(entity_type)


        for json_relation in json_line['relationMentions']:

            label = json_relation['label']
            relationType.add(label)

    fp.close()



    print(entityType)
    print(relationType)

def preprocess(data, fileName, name):
    fp = open(fileName, 'r')

    root_dir = fileName[0:fileName.rfind('.')]
    preprocess_dir = os.path.join(root_dir, "preprocessed")
    # if os.path.exists(preprocess_dir):
    #     shutil.rmtree(preprocess_dir)
    #     os.makedirs(preprocess_dir)
    # else:
    #     os.makedirs(preprocess_dir)

    if os.path.exists(preprocess_dir) == False:
        os.makedirs(preprocess_dir)

    index = 1

    max_len = data.MAX_SENTENCE_LENGTH if data.MAX_SENTENCE_LENGTH > data.max_seq_len else data.max_seq_len

    for line in tqdm(fp.readlines()):



        json_line = json.loads(line)

        # if index < 230225:
        #     index += 1
        #     continue

        articleId = json_line['articleId'][json_line['articleId'].rfind('/')+1:]
        sentId = json_line['sentId']
        sentText = json_line['sentText']
        tmp_tokens = token_from_sent(sentText, 0)

        if name != "test" and len(tmp_tokens) > max_len:
            index += 1
            continue

        # here we use one sentence as a df_doc
        df_doc = pd.DataFrame()  # contains token-level information
        df_tokens = pd.DataFrame(tmp_tokens, columns=['text', 'postag', 'start', 'end'])
        df_sent_id = pd.DataFrame([sentId]*len(tmp_tokens), columns = ['sent_idx'])
        df_comb = pd.concat([df_tokens, df_sent_id], axis=1)
        df_doc = pd.concat([df_doc, df_comb])
        df_doc.index = range(df_doc.shape[0])

        # entity
        anno_data = []
        all_entity_id = set()
        last_entity_end = 0
        for json_entity in json_line['entityMentions']:
            entity_id = json_entity['start'] # I don't know the meaning of start
            if entity_id in all_entity_id:
                raise RuntimeError('file {}, sentence {}, entity {}, reduplicative entity id'.format(articleId, sentId, entity_id))
            else:
                all_entity_id.add(entity_id)

            entity_type = json_entity['label']
            entity_text = json_entity['text']
            entity_sentId = sentId

            # pattern = re.compile(r'^(Bobby)\s+|\s+(Bobby)\s+|\s+(Bobby)$')
            # string = 'Bobby fdafd Bobbys fdkj Bobby   Bobby'
            # match = pattern.search(string, 0)
            # while match:
            #     if match.group(1):
            #         start = match.start(1)
            #         end = match.end(1)
            #         text = match.group(1)
            #     elif match.group(2):
            #         start = match.start(2)
            #         end = match.end(2)
            #         text = match.group(2)
            #     else:
            #         start = match.start(3)
            #         end = match.end(3)
            #         text = match.group(3)
            #
            #     match = pattern.search(string, end)

            # last_entity_start = -1
            # last_entity_end = -1
            # for old_entity_id, old_start, old_end, old_entity_text, old_entity_type, old_entity_sentId, old_tf_start, old_tf_end in anno_data:
            #     if entity_text == old_entity_text:
            #         last_entity_start = old_start
            #         last_entity_end = old_end

            # pattern = re.compile(r'^(' + entity_text + r')\s+|\s+(' + entity_text + r')\s+|\s+(' + entity_text + r')$')
            # if last_entity_start != -1:
            #     # we get a entity with same text
            #     match = pattern.search(new_sentText, last_entity_end)
            # else:
            #     match = pattern.search(new_sentText, 0)
            #
            # if match == None:
            #     raise RuntimeError('file {}, sentence {}, entity {}, match none'.format(articleId, sentId, entity_id))
            #
            # if match.group(1):
            #     start = match.start(1)
            #     end = match.end(1)
            # elif match.group(2):
            #     start = match.start(2)
            #     end = match.end(2)
            # else:
            #     start = match.start(3)
            #     end = match.end(3)

            # pattern = re.compile(r'^('+entity_text+r')\s+|\s+('+entity_text+r')\s+|\s+('+entity_text+r')$')
            # match = pattern.search(sentText, 0)
            # if match:
            #     if match.group(1):
            #         start = match.start(1)
            #         end = match.end(1)
            #     elif match.group(2):
            #         start = match.start(2)
            #         end = match.end(2)
            #     else:
            #         start = match.start(3)
            #         end = match.end(3)
            #     match = pattern.search(sentText, end)
            #     if match:
            #         print(
            #             'file {}, sentence {}, entity {}, multi entity mentions'.format(articleId, sentId, entity_id))
            # else:
            #     raise RuntimeError(
            #         'file {}, sentence {}, entity {}, no entity mentions'.format(articleId, sentId, entity_id))


            # if last_entity_start != -1:
            #     # we get a entity with same text
            #     start = sentText.find(entity_text, last_entity_end)
            # else:
            #     start = sentText.find(entity_text, 0)

            current_start = last_entity_end

            while current_start < len(sentText):

                start = sentText.find(entity_text, current_start)
                if start == -1:
                    raise RuntimeError('file {}, sentence {}, entity {}, cannot find entity mentions'.format(articleId, sentId, entity_id))
                end = start + len(entity_text)
                current_start = end

                while current_start < len(sentText) and sentText[current_start].isalpha(): # non complete match
                    start = sentText.find(entity_text, current_start)
                    if start == -1:
                        raise RuntimeError('file {}, sentence {}, entity {}, cannot find entity mentions'.format(articleId, sentId, entity_id))
                    end = start + len(entity_text)
                    current_start = end

                df_sentence = df_doc[(df_doc['sent_idx'] == entity_sentId)]
                tf_start = -1
                tf_end = -1
                token_num = df_sentence.shape[0]

                for tf_idx in range(token_num):
                    token = df_sentence.iloc[tf_idx]

                    if token['start'] == start:
                        tf_start = tf_idx
                    if token['end'] == end:
                        tf_end = tf_idx

                if tf_start != -1 and tf_end != -1:
                    break

            if tf_start == -1 or tf_end == -1:
                raise RuntimeError('file {}, sentence {}, entity {}, not found tf_start or tf_end'.format(articleId, sentId, entity_id))

            last_entity_end = current_start

            # start = sentText.find(entity_text, 0) # assume there is only one mention in the sentence
            # end = start + len(entity_text)
            # if sentText.find(entity_text, end) != -1:
            #     raise RuntimeError(
            #         'file {}, sentence {}, entity {}, multi entity mentions'.format(articleId, sentId, entity_id))





            # if tf_start == -1 or tf_end == -1: # give a second chance to find
            #
            #     while end < len(sentText):
            #         start = sentText.find(entity_text, last_entity_end)
            #         end = start + len(entity_text)
            #         if end < len(sentText):
            #             while sentText[end].isalpha():
            #                 start = sentText.find(entity_text, end)
            #                 end = start + len(entity_text)
            #
            #         if start == -1:
            #             raise RuntimeError(
            #                 'file {}, sentence {}, entity {}, cannot find entity mentions'.format(articleId, sentId,
            #                                                                                       entity_id))
            #
            #
            #         df_sentence = df_doc[(df_doc['sent_idx'] == entity_sentId)]
            #         tf_start = -1
            #         tf_end = -1
            #         token_num = df_sentence.shape[0]
            #
            #         for tf_idx in range(token_num):
            #             token = df_sentence.iloc[tf_idx]
            #
            #             if token['start'] == start:
            #                 tf_start = tf_idx
            #             if token['end'] == end:
            #                 tf_end = tf_idx
            #
            #         if tf_start != -1 and tf_end != -1:
            #             break
            #         else:
            #             last_entity_end = end
            #
            # if tf_start == -1 or tf_end == -1:
            #     raise RuntimeError('file {}, sentence {}, entity {}, not found tf_start or tf_end'.format(articleId, sentId, entity_id))
            #

            anno_data.append([entity_id, start, end, entity_text, entity_type, entity_sentId, tf_start, tf_end])

        df_entity = pd.DataFrame(anno_data, columns=['id', 'start', 'end', 'text', 'type', 'sent_idx', 'tf_start',
                                                     'tf_end'])  # contains entity information
        df_entity = df_entity.sort_values('start')
        df_entity.index = range(df_entity.shape[0])


        # relation
        relation_data = []
        relationID = 1
        for json_relation in json_line['relationMentions']:
            em1Text = json_relation['em1Text']
            em2Text = json_relation['em2Text']
            label = json_relation['label']

            # entity1 = df_entity[(df_entity['text'] == em1Text)]
            # if entity1.shape[0] != 1:
            #     raise RuntimeError('file {}, sentence {}, entity {}, argument error'.format(articleId, sentId, em1Text))
            #
            # entity2 = df_entity[(df_entity['text'] == em2Text)]
            # if entity2.shape[0] != 1:
            #     raise RuntimeError('file {}, sentence {}, entity {}, argument error'.format(articleId, sentId, em2Text))
            #
            # relation_data.append([relationID, label, entity1.iloc[0]['id'], entity2.iloc[0]['id']])

            relation_data.append([relationID, label, em1Text, em2Text])

            relationID += 1

        df_relation = pd.DataFrame(relation_data, columns=['id', 'type', 'entity1_text', 'entity2_text'])




        df_doc.to_pickle(os.path.join(preprocess_dir, '{}.token'.format(index)))
        df_entity.to_pickle(os.path.join(preprocess_dir, '{}.entity'.format(index)))
        df_relation.to_pickle(os.path.join(preprocess_dir, '{}.relation'.format(index)))

        index += 1

    fp.close()

pattern = re.compile(r'[-_\.]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text

def my_tokenize(txt):
    return re.split(r'\s+', txt)
    # tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    # tokens2 = []
    # for token1 in tokens1:
    #     token2 = my_split(token1)
    #     tokens2.extend(token2)
    # return tokens2

def text_tokenize_and_postagging(txt, sent_start):
    new_txt = txt.strip()
    new_txt = re.sub(r"\'|\"", ',', new_txt)

    tokens=my_tokenize(new_txt)

    pos_tags = nltk.pos_tag(tokens)

    result = []
    offset = 0
    for token, pos_tag in pos_tags:
        offset = new_txt.find(token, offset)
        result.append((token, pos_tag, offset+sent_start, offset+len(token)+sent_start))
        offset += len(token)
    return result

def token_from_sent(txt, sent_start):
    #return [token for token in text_tokenize(txt, sent_start)]
    #return [token for token in text_tokenize_and_postagging(txt, sent_start)]
    return text_tokenize_and_postagging(txt, sent_start)

def loadPreprocessData(fileName):
    root_dir = fileName[0:fileName.rfind('.')]
    preprocess_dir = os.path.join(root_dir, "preprocessed")

    files = list(set([f[0:f.find('.')] for f in os.listdir(preprocess_dir)]))
    files.sort()

    df_all_doc = []
    df_all_entity = []
    df_all_relation = []
    all_name = []
    for f in files:
        df_all_doc.append(pd.read_pickle(os.path.join(preprocess_dir,f+'.token')))
        df_all_entity.append(pd.read_pickle(os.path.join(preprocess_dir,f+'.entity')))
        df_all_relation.append(pd.read_pickle(os.path.join(preprocess_dir,f+'.relation')))
        all_name.append(f)

    logging.info("load preprocessed data complete in {}".format(preprocess_dir))

    return df_all_doc, df_all_entity, df_all_relation, all_name


def generateData(tokens, entitys, names, output_file):

    f = open(output_file, 'w')

    for i in tqdm(range(len(names))):
        sent_token = tokens[i]
        sent_entity = entitys[i]

        for _, token in sent_token.iterrows():
            word = token['text']
            pos = token['postag']
            cap = my_utils.featureCapital(word)
            label = getLabel(token['start'], token['end'], sent_entity)
            f.write("{} [Cap]{} [POS]{} {}\n".format(word, cap, pos, label))
            # try :
            #     f.write("{} [Cap]{} [POS]{} {}\n".format(word, cap, pos, label))
            #     # s = u"{} ".format(u"puréeing")
            #     # f.write(s)
            # except Exception:
            #     print(u"non asc word {}, label {}".format(word, label))
            #     word = 'x'*len(word)
            #     f.write("{} [Cap]{} [POS]{} {}\n".format(word, cap, pos, label))
            # f.write(u"{} [Cap]{} [POS]{} {}\n".format(word, cap, pos, label))

        f.write("\n")

    f.close()


ENTITY_TYPE = set(['PERSON', 'LOCATION', 'ORGANIZATION'])
def getLabel(start, end, sent_entity):
    match = ""
    for index, entity in sent_entity.iterrows():
        if start == entity['start'] and end == entity['end'] : # S
            match = "S"
            break
        elif start == entity['start'] and end != entity['end'] : # B
            match = "B"
            break
        elif start != entity['start'] and end == entity['end'] : # E
            match = "E"
            break
        elif start > entity['start'] and end < entity['end']:  # M
            match = "M"
            break

    if match != "" and sent_entity.loc[index]['type'] in ENTITY_TYPE:
        return match+"-"+sent_entity.loc[index]['type']
    else:
        return "O"
    

def build_re_feature_alphabets(data, tokens, entities, relations):

    entity_type_alphabet = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']]
    entity_alphabet = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']]
    relation_alphabet = data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']]
    token_num_alphabet = data.re_feature_alphabets[data.re_feature_name2id['[TOKEN_NUM]']]
    entity_num_alphabet = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_NUM]']]
    position_alphabet = data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']]

    for i, doc_token in enumerate(tokens):

        doc_entity = entities[i]
        doc_relation = relations[i]

        sentence = doc_token

        entities_in_sentence = doc_entity
        for _, entity in entities_in_sentence.iterrows():
            entity_type_alphabet.add(entity['type'])
            tk_idx = entity['tf_start']
            while tk_idx <= entity['tf_end']:
                entity_alphabet.add(
                    my_utils1.normalizeWord(sentence.iloc[tk_idx, 0]))  # assume 'text' is in 0 column
                tk_idx += 1



        for _, relation in doc_relation.iterrows():
            if relation['type'] != 'None':
                relation_alphabet.add(relation['type'])


    for i in range(data.max_seq_len):
        token_num_alphabet.add(i)
        entity_num_alphabet.add(i)
        position_alphabet.add(i)
        position_alphabet.add(-i)


    for idx in range(data.re_feature_num):
        data.re_feature_alphabet_sizes[idx] = data.re_feature_alphabets[idx].size()


def generate_re_instance(data, name, tokens, entities, relations, names):
    data.fix_re_alphabet()
    if name == "train":
        data.re_train_X, data.re_train_Y = getRelationInstance(tokens, entities, relations, names, data)
    elif name == "dev":
        data.re_dev_X, data.re_dev_Y = getRelationInstance(tokens, entities, relations, names, data)
    elif name == "test":
        data.re_test_X, data.re_test_Y = getRelationInstance(tokens, entities, relations, names, data)
    else:
        print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


def getRelationInstance(tokens, entities, relations, names, data):
    X = []
    Y = []
    cnt_neg = 0

    for i in tqdm(range(len(relations))):

        doc_relation = relations[i]
        doc_token = tokens[i]
        doc_entity = entities[i] # entity are sorted by start offset
        doc_name = names[i]

        row_num = doc_entity.shape[0]

        for latter_idx in range(row_num):

            for former_idx in range(row_num):

                if former_idx < latter_idx:

                    former = doc_entity.iloc[former_idx]
                    latter = doc_entity.iloc[latter_idx]

                    if former['text'] == latter['text']:
                        continue


                    gold_relations = doc_relation[
                        (
                                ((doc_relation['entity1_text'] == former['text']) & (
                                            doc_relation['entity2_text'] == latter['text']))
                                |
                                ((doc_relation['entity1_text'] == latter['text']) & (
                                            doc_relation['entity2_text'] == former['text']))
                        )
                    ]
                    # if gold_relations.shape[0] == 0:
                    #     raise RuntimeError("{}: entity {} and {} has strange relations".format(doc_name, former['id'], latter['id']))


                    context_token = doc_token
                    former_tf_start, former_tf_end = former['tf_start'], former['tf_end']
                    latter_tf_start, latter_tf_end = latter['tf_start'], latter['tf_end']

                    if context_token.shape[0] > data.max_seq_len:
                        # truncate
                        logging.debug("exceed max_seq_len {} {}".format(doc_name, context_token.shape[0]))
                        context_token = context_token.iloc[:data.max_seq_len]


                    words = []
                    postags = []
                    cap = []
                    chars = []
                    positions1 = []
                    positions2 = []
                    former_token = []
                    latter_token = []
                    i = 0
                    for _, token in context_token.iterrows():
                        if data.number_normalized:
                            word = normalize_word(token['text'])
                        else:
                            word = token['text']
                        entity_word = my_utils1.normalizeWord(token['text'])
                        words.append(data.word_alphabet.get_index(word))
                        postags.append(data.feature_alphabets[data.feature_name2id['[POS]']].get_index(token['postag']))
                        cap.append(data.feature_alphabets[data.feature_name2id['[Cap]']].get_index(str(my_utils.featureCapital(token['text']))))
                        char_for1word = []
                        for char in word:
                            char_for1word.append(data.char_alphabet.get_index(char))
                        chars.append(char_for1word)

                        if i < former_tf_start:
                            positions1.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(former_tf_start - i))

                        elif i > former_tf_end:
                            positions1.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(former_tf_end - i))

                        else:
                            positions1.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(0))
                            former_token.append(data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(entity_word))

                        if i < latter_tf_start:
                            positions2.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(latter_tf_start - i))

                        elif i > latter_tf_end:
                            positions2.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(latter_tf_end - i))

                        else:
                            positions2.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(0))
                            latter_token.append(data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(entity_word))

                        i += 1

                    if len(former_token) == 0: # truncated part contains entity, so we have to use the text in doc_entity
                        splitted = my_utils.my_tokenize(former['text'])
                        for s in splitted:
                            s = s.strip()
                            if s != "":
                                former_token.append(data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(my_utils1.normalizeWord(s)))
                    if len(latter_token) == 0:
                        splitted = my_utils.my_tokenize(latter['text'])
                        for s in splitted:
                            s = s.strip()
                            if s != "":
                                latter_token.append(data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(my_utils1.normalizeWord(s)))

                    assert len(former_token)>0
                    assert len(latter_token)>0


                    features = {'tokens': words, 'postag': postags, 'cap': cap, 'char': chars, 'positions1': positions1, 'positions2': positions2}
                    features['e1_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(former['type'])
                    features['e2_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(latter['type'])
                    features['e1_token'] = former_token
                    features['e2_token'] = latter_token


                    features['tok_num_betw'] = data.re_feature_alphabets[data.re_feature_name2id['[TOKEN_NUM]']].get_index(latter['tf_start']-former['tf_end'])

                    entity_between = doc_entity[((doc_entity['start']>=former['end']) & (doc_entity['end']<=latter['start']))]
                    features['et_num'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_NUM]']].get_index(entity_between.shape[0])

                    X.append(features)

                    gold_answer = '</unk>'
                    for _, gold_relation in gold_relations.iterrows():
                        if gold_relation['type']!='None':
                            gold_answer = gold_relation['type']
                            break

                    Y.append(data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']].get_index(gold_answer))
                    if gold_answer == '</unk>':
                        cnt_neg += 1

                    # if gold_relations.iloc[0]['type']=='None' and gold_relations.iloc[1]['type']=='None':
                    #     Y.append(data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']].get_index('</unk>'))
                    #     cnt_neg += 1
                    # else:
                    #     gold_answer = gold_relations.iloc[0]['type'] if gold_relations.iloc[0]['type']!='None' else gold_relations.iloc[1]['type']
                    #     Y.append(data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']].get_index(gold_answer))


    neg = 100.0*cnt_neg/len(Y)

    logging.info("positive instance {}%, negative instance {}%".format(100-neg, neg))
    return X, Y

