

from torch.utils.data import Dataset
import torch
import numpy as np
import logging
from tqdm import tqdm
import os
import shutil
import math
import pandas as pd
import re
import nltk
from utils.data import data

# lowercased, number to 0, punctuation to #
ENG_PUNC = set(['`','~','!','@','#','$','%','&','*','(',')','-','_','+','=','{',
                '}','|','[',']','\\',':',';','\'','"','<','>',',','.','?','/'])

DIGIT = set(['0','1','2','3','4','5','6','7','8','9'])
def normalizeWord(word, cased=False):
    newword = ''
    for ch in word:
        if ch in DIGIT:
            newword = newword + '0'
        elif ch in ENG_PUNC:
            newword = newword + '#'
        else:
            newword = newword + ch

    if not cased:
        newword = newword.lower()
    return newword


pattern = re.compile(r'[-_/]+')

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
    tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    tokens2 = []
    for token1 in tokens1:
        token2 = my_split(token1)
        tokens2.extend(token2)
    return tokens2

# relation constraints
# 'do': set(['Drug Dose', 'Dose Dose']),
# 'fr': set(['Drug Frequency', 'Frequency Frequency']),
# 'manner/route': set(['Drug Route', 'Route Route']),
# 'Drug_By Patient': set(['Drug By Patient']),
# 'severity_type': set(['Indication Severity', 'ADE Severity', 'SSLIF Severity', 'Severity Severity']),
# 'adverse': set(['Drug ADE', 'SSLIF ADE', 'ADE ADE']),
# 'reason': set(['Drug Indication', 'Indication Indication']),
# 'Drug_By Physician': set(['Drug By Physician']),
# 'du': set(['Duration Duration', 'Drug Duration'])


def relationConstraint1(relation_type, type1, type2):

    if relation_type=='do':
        if (type1 == 'Drug' and type2 == 'Dose') or (type1 == 'Dose' and type2 == 'Drug') or (
                type1 == 'Dose' and type2 == 'Dose'):
            return True
        else:
            return False

    elif relation_type=='fr':
        if (type1 == 'Drug' and type2 == 'Frequency') or (type1 == 'Frequency' and type2 == 'Drug') or (
                type1 == 'Frequency' and type2 == 'Frequency'):
            return True
        else:
            return False
    elif relation_type=='manner/route':
        if (type1 == 'Drug' and type2 == 'Route') or (type1 == 'Route' and type2 == 'Drug') or (
                type1 == 'Route' and type2 == 'Route'):
            return True
        else:
            return False
    elif relation_type=='Drug_By Patient':
        if (type1 == 'Drug By' and type2 == 'Patient') or (type1 == 'Patient' and type2 == 'Drug By'):
            return True
        else:
            return False
    # cardio begin
    elif relation_type=='severity_type':
        if (type1 == 'Indication' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'Indication') or \
                (type1 == 'ADE' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'ADE') or \
                (type1 == 'SSLIF' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'SSLIF') \
                or (type1 == 'Bleeding' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'Bleeding') \
                or (type1 == 'BleedingLabEval' and type2 == 'Severity') or (type1 == 'Severity' and type2 == 'BleedingLabEval') \
                or (type1 == 'Severity' and type2 == 'Severity'):
            return True
        else:
            return False
    # cardio end
    elif relation_type=='adverse':
        if (type1 == 'Drug' and type2 == 'ADE') or (type1 == 'ADE' and type2 == 'Drug') or \
                (type1 == 'SSLIF' and type2 == 'ADE') or (type1 == 'ADE' and type2 == 'SSLIF') \
                or (type1 == 'ADE' and type2 == 'ADE'):
            return True
        else:
            return False
    elif relation_type=='reason':
        if (type1 == 'Drug' and type2 == 'Indication') or (type1 == 'Indication' and type2 == 'Drug') or (
                type1 == 'Indication' and type2 == 'Indication'):
            return True
        else:
            return False
    elif relation_type=='Drug_By Physician':
        if (type1 == 'Drug By' and type2 == 'Physician') or (type1 == 'Physician' and type2 == 'Drug By'):
            return True
        else:
            return False
    elif relation_type=='du':
        if (type1 == 'Drug' and type2 == 'Duration') or (type1 == 'Duration' and type2 == 'Drug') or (
                type1 == 'Duration' and type2 == 'Duration'):
            return True
        else:
            return False
    # cardio begin
    elif relation_type=='Bleeding_BleedingAnatomicSite':
        if (type1 == 'Bleeding' and type2 == 'BleedingAnatomicSite') or (type1 == 'BleedingAnatomicSite' and type2 == 'Bleeding'):
            return True
        else:
            return False
    # cardio end
    else:
        raise RuntimeError("unknown relation type {}".format(relation_type))


def relationConstraint_chapman(type1, type2): # determine whether the constraint are satisfied, non-directional

    if (type1 == 'Drug' and type2 == 'Dose'):
        return 1
    elif (type1 == 'Dose' and type2 == 'Drug'):
        return -1
    elif (type1 == 'Drug' and type2 == 'Frequency'):
        return 1
    elif (type1 == 'Frequency' and type2 == 'Drug'):
        return -1
    elif (type1 == 'Drug' and type2 == 'Route'):
        return 1
    elif (type1 == 'Route' and type2 == 'Drug'):
        return -1
    elif (type1 == 'Drug By' and type2 == 'Patient'):
        return 1
    elif (type1 == 'Patient' and type2 == 'Drug By'):
        return -1
    elif (type1 == 'Indication' and type2 == 'Severity') or (type1 == 'ADE' and type2 == 'Severity') or (type1 == 'SSLIF' and type2 == 'Severity'):
        return 1
    elif (type1 == 'Severity' and type2 == 'Indication') or (type1 == 'Severity' and type2 == 'ADE') or (type1 == 'Severity' and type2 == 'SSLIF'):
        return -1
    elif (type1 == 'Drug' and type2 == 'ADE'):
        return 1
    elif (type1 == 'ADE' and type2 == 'Drug'):
        return -1
    elif (type1 == 'Drug' and type2 == 'Indication'):
        return 1
    elif (type1 == 'Indication' and type2 == 'Drug'):
        return -1
    elif (type1 == 'Drug By' and type2 == 'Physician'):
        return 1
    elif (type1 == 'Physician' and type2 == 'Drug By'):
        return -1
    elif (type1 == 'Drug' and type2 == 'Duration'):
        return 1
    elif (type1 == 'Duration' and type2 == 'Drug'):
        return -1
    # cardio begin
    # cardio annotation are not all consistent with made
    elif (type1 == 'Bleeding' and type2 == 'Severity'):
        return 1
    elif (type1 == 'Severity' and type2 == 'Bleeding'):
        return -1
    elif (type1 == 'BleedingLabEval' and type2 == 'Severity'):
        return 1
    elif (type1 == 'Severity' and type2 == 'BleedingLabEval'):
        return -1
    elif (type1 == 'Bleeding' and type2 == 'BleedingAnatomicSite'):
        return 1
    elif (type1 == 'BleedingAnatomicSite' and type2 == 'Bleeding'):
        return -1
    # cardio end
    else:
        return 0




# truncate before feature
def getRelationInstance2(tokens, entities, relations, names, word_vocab, postag_vocab,
                         relation_vocab, entity_type_vocab, entity_vocab,
                         position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab):
    X = []
    Y = []
    other = [] # other is used for outputing results, it's usually used for test set
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


                    if math.fabs(latter['sent_idx']-former['sent_idx']) >= data.sent_window:
                        continue

                    # for double annotation, we don't generate instances
                    if former['start']==latter['start'] and former['end']==latter['end']:
                        continue

                    #type_constraint = relationConstraint(former['type'], latter['type'])
                    type_constraint = relationConstraint_chapman(former['type'], latter['type'])
                    if type_constraint == 0:
                        continue

                    gold_relations = doc_relation[
                        (
                                ((doc_relation['entity1_id'] == former['id']) & (
                                            doc_relation['entity2_id'] == latter['id']))
                                |
                                ((doc_relation['entity1_id'] == latter['id']) & (
                                            doc_relation['entity2_id'] == former['id']))
                        )
                    ]
                    if gold_relations.shape[0] > 1:
                        #raise RuntimeError("the same entity pair has more than one relations")
                        logging.debug("entity {} and {} has more than one relations".format(former['id'], latter['id']))
                        continue

                    # here we retrieve all the sentences inbetween two entities, sentence of former, sentence ..., sentence of latter
                    sent_idx = former['sent_idx']
                    context_token = pd.DataFrame(columns=doc_token.columns)
                    base = 0
                    former_tf_start, former_tf_end = -1, -1
                    latter_tf_start, latter_tf_end = -1, -1
                    while sent_idx <= latter['sent_idx']:
                        sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]

                        if former['sent_idx'] == sent_idx:
                            former_tf_start, former_tf_end = base+former['tf_start'], base+former['tf_end']
                        if latter['sent_idx'] == sent_idx:
                            latter_tf_start, latter_tf_end = base+latter['tf_start'], base+latter['tf_end']

                        context_token = context_token.append(sentence, ignore_index=True)

                        base += len(sentence['text'])
                        sent_idx += 1

                    if context_token.shape[0] > data.max_seq_len:
                        # truncate
                        logging.debug("exceed max_seq_len {} {}".format(doc_name, context_token.shape[0]))
                        context_token = context_token.iloc[:data.max_seq_len]


                    words = []
                    postags = []
                    positions1 = []
                    positions2 = []
                    former_token = []
                    latter_token = []
                    i = 0
                    for _, token in context_token.iterrows():
                        word = normalizeWord(token['text'])
                        words.append(word_vocab.lookup(word))
                        postags.append(postag_vocab.lookup(token['postag']))

                        if i < former_tf_start:
                            positions1.append(position_vocab1.lookup(former_tf_start - i))
                        elif i > former_tf_end:
                            positions1.append(position_vocab1.lookup(former_tf_end - i))
                        else:
                            positions1.append(position_vocab1.lookup(0))
                            former_token.append(entity_vocab.lookup(word))

                        if i < latter_tf_start:
                            positions2.append(position_vocab2.lookup(latter_tf_start - i))
                        elif i > latter_tf_end:
                            positions2.append(position_vocab2.lookup(latter_tf_end - i))
                        else:
                            positions2.append(position_vocab2.lookup(0))
                            latter_token.append(entity_vocab.lookup(word))

                        i += 1

                    if len(former_token) == 0: # truncated part contains entity, so we have to use the text in doc_entity
                        # splitted = re.split(r"\s+| +|[\(\)\[\]\-_,]+", former['text'])
                        splitted = my_tokenize(former['text'])
                        for s in splitted:
                            s = s.strip()
                            if s != "":
                                former_token.append(entity_vocab.lookup(normalizeWord(s)))
                    if len(latter_token) == 0:
                        #splitted = re.split(r"\s+| +|[\(\)\[\]\-_,]+", latter['text'])
                        splitted = my_tokenize(latter['text'])
                        for s in splitted:
                            s = s.strip()
                            if s != "":
                                latter_token.append(entity_vocab.lookup(normalizeWord(s)))

                    assert len(former_token)>0
                    assert len(latter_token)>0


                    features = {'tokens': words, 'postag': postags, 'positions1': positions1, 'positions2': positions2}
                    if type_constraint == 1:
                        features['e1_type'] = entity_type_vocab.lookup(former['type'])
                        features['e2_type'] = entity_type_vocab.lookup(latter['type'])
                        features['e1_token'] = former_token
                        features['e2_token'] = latter_token
                    else:
                        features['e1_type'] = entity_type_vocab.lookup(latter['type'])
                        features['e2_type'] = entity_type_vocab.lookup(former['type'])
                        features['e1_token'] = latter_token
                        features['e2_token'] = former_token

                    features['tok_num_betw'] = tok_num_betw_vocab.lookup(latter['tf_start']-former['tf_end'])

                    entity_between = doc_entity[((doc_entity['start']>=former['end']) & (doc_entity['end']<=latter['start']))]
                    features['et_num'] = et_num_vocab.lookup(entity_between.shape[0])

                    X.append(features)

                    if gold_relations.shape[0] == 0:
                        Y.append(relation_vocab.lookup('<unk>'))
                        cnt_neg += 1
                    else:
                        gold_answer = gold_relations.iloc[0]['type']
                        Y.append(relation_vocab.lookup(gold_answer))

                    other_info = {}
                    other_info['doc_name'] = doc_name
                    other_info['former_id'] = former['id']
                    other_info['latter_id'] = latter['id']
                    other.append(other_info)




    neg = 100.0*cnt_neg/len(Y)

    logging.info("positive instance {}%, negative instance {}%".format(100-neg, neg))
    return X, Y, other


def getRelationInstanceForOneDoc(doc_token, entities, doc_name, word_vocab, postag_vocab,
                         relation_vocab, entity_type_vocab, entity_vocab,
                         position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab):
    X = []
    other = []

    row_num = len(entities)

    for latter_idx in range(row_num):

        for former_idx in range(row_num):

            if former_idx < latter_idx:

                former = entities[former_idx]
                latter = entities[latter_idx]


                if math.fabs(latter.sent_idx-former.sent_idx) >= data.sent_window:
                    continue

                # for double annotation, we don't generate instances
                if former.start==latter.start and former.end==latter.end:
                    continue

                #type_constraint = relationConstraint(former['type'], latter['type'])
                type_constraint = relationConstraint_chapman(former.type, latter.type)
                if type_constraint == 0:
                    continue

                # here we retrieve all the sentences inbetween two entities, sentence of former, sentence ..., sentence of latter
                sent_idx = former.sent_idx
                context_token = pd.DataFrame(columns=doc_token.columns)
                base = 0
                former_tf_start, former_tf_end = -1, -1
                latter_tf_start, latter_tf_end = -1, -1
                while sent_idx <= latter.sent_idx:
                    sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]

                    if former.sent_idx == sent_idx:
                        former_tf_start, former_tf_end = base+former.tf_start, base+former.tf_end
                    if latter.sent_idx == sent_idx:
                        latter_tf_start, latter_tf_end = base+latter.tf_start, base+latter.tf_end

                    context_token = context_token.append(sentence, ignore_index=True)

                    base += len(sentence['text'])
                    sent_idx += 1

                if context_token.shape[0] > data.max_seq_len:
                    # truncate
                    logging.debug("exceed max_seq_len {} {}".format(doc_name, context_token.shape[0]))
                    context_token = context_token.iloc[:data.max_seq_len]


                words = []
                postags = []
                positions1 = []
                positions2 = []
                former_token = []
                latter_token = []
                i = 0
                for _, token in context_token.iterrows():
                    word = normalizeWord(token['text'])
                    words.append(word_vocab.lookup(word))
                    postags.append(postag_vocab.lookup(token['postag']))

                    if i < former_tf_start:
                        positions1.append(position_vocab1.lookup(former_tf_start - i))
                    elif i > former_tf_end:
                        positions1.append(position_vocab1.lookup(former_tf_end - i))
                    else:
                        positions1.append(position_vocab1.lookup(0))
                        former_token.append(entity_vocab.lookup(word))

                    if i < latter_tf_start:
                        positions2.append(position_vocab2.lookup(latter_tf_start - i))
                    elif i > latter_tf_end:
                        positions2.append(position_vocab2.lookup(latter_tf_end - i))
                    else:
                        positions2.append(position_vocab2.lookup(0))
                        latter_token.append(entity_vocab.lookup(word))

                    i += 1

                if len(former_token) == 0: # truncated part contains entity, so we have to use the text in doc_entity
                    # splitted = re.split(r"\s+| +|[\(\)\[\]\-_,]+", former['text'])
                    splitted = my_tokenize(former.text)
                    for s in splitted:
                        s = s.strip()
                        if s != "":
                            former_token.append(entity_vocab.lookup(normalizeWord(s)))
                if len(latter_token) == 0:
                    #splitted = re.split(r"\s+| +|[\(\)\[\]\-_,]+", latter['text'])
                    splitted = my_tokenize(latter.text)
                    for s in splitted:
                        s = s.strip()
                        if s != "":
                            latter_token.append(entity_vocab.lookup(normalizeWord(s)))

                assert len(former_token)>0
                assert len(latter_token)>0


                features = {'tokens': words, 'postag': postags, 'positions1': positions1, 'positions2': positions2}
                if type_constraint == 1:
                    features['e1_type'] = entity_type_vocab.lookup(former.type)
                    features['e2_type'] = entity_type_vocab.lookup(latter.type)
                    features['e1_token'] = former_token
                    features['e2_token'] = latter_token
                else:
                    features['e1_type'] = entity_type_vocab.lookup(latter.type)
                    features['e2_type'] = entity_type_vocab.lookup(former.type)
                    features['e1_token'] = latter_token
                    features['e2_token'] = former_token

                features['tok_num_betw'] = tok_num_betw_vocab.lookup(latter.tf_start-former.tf_end)

                entity_between = getEntitiesBetween(former, latter, entities)
                features['et_num'] = et_num_vocab.lookup(len(entity_between))

                X.append(features)

                other.append((former, latter))

    return X, other

def getEntitiesBetween(former, latter, entities):
    results = []
    for entity in entities:
        if entity.start >= former.end and entity.end <= latter.start:
            results.append(entity)

    return results

def getEntities(id, entities):
    for entity in entities:
        if id == entity.id:
            return entity

    return None


class RelationDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])



def sorted_collate(batch):
    return my_collate(batch, sort=True)


def unsorted_collate(batch):
    return my_collate(batch, sort=False)


def my_collate(batch, sort):
    if len(batch[0]) ==2 : # I made a bad hypothesis
        x, y = zip(*batch)
    else:
        x = batch
        y = None

    x2, x1, y = pad(x, y, data.pad_idx, sort)

    if torch.cuda.is_available():
        for i, _ in enumerate(x2):
            if x2[i] is not None:
                x2[i] = x2[i].cuda(data.HP_gpu)
        for i, _ in enumerate(x1):
            if x1[i] is not None:
                x1[i] = x1[i].cuda(data.HP_gpu)
        if y is not None:
            y = y.cuda(data.HP_gpu)
    return x2, x1, y


def pad(x, y, eos_idx, sort):
    tokens = [s['tokens'] for s in x]
    postag = [s['postag'] for s in x]
    positions1 = [s['positions1'] for s in x]
    positions2 = [s['positions2'] for s in x]
    e1_type = [s['e1_type'] for s in x]
    e2_type = [s['e2_type'] for s in x]
    e1_token = [s['e1_token'] for s in x]
    e2_token = [s['e2_token'] for s in x]
    tok_num_betw = [s['tok_num_betw'] for s in x]
    et_num = [s['et_num'] for s in x]

    lengths = [len(row) for row in tokens]
    max_len = max(lengths)
    # cnnrnn
    if data.feature_extractor == 'cnn':
        max_len = max(max_len, 5)

    # features
    tokens = pad_sequence(tokens, max_len, eos_idx)
    lengths = torch.LongTensor(lengths)

    postag = pad_sequence(postag, max_len, eos_idx)

    positions1 = pad_sequence(positions1, max_len, eos_idx)
    positions2 = pad_sequence(positions2, max_len, eos_idx)

    e1_length = [len(row) for row in e1_token]
    max_e1_length = max(e1_length)
    e1_token = pad_sequence(e1_token, max_e1_length, eos_idx)
    e1_length = torch.LongTensor(e1_length)

    e2_length = [len(row) for row in e2_token]
    max_e2_length = max(e2_length)
    e2_token = pad_sequence(e2_token, max_e2_length, eos_idx)
    e2_length = torch.LongTensor(e2_length)

    e1_type = torch.LongTensor(e1_type)
    e2_type = torch.LongTensor(e2_type)

    tok_num_betw = torch.LongTensor(tok_num_betw)

    et_num = torch.LongTensor(et_num)

    if y:
        y = torch.LongTensor(y).view(-1)

    if sort:
        # sort by length
        sort_len, sort_idx = lengths.sort(0, descending=True)
        tokens = tokens.index_select(0, sort_idx)

        postag = postag.index_select(0, sort_idx)

        positions1 = positions1.index_select(0, sort_idx)
        positions2 = positions2.index_select(0, sort_idx)

        e1_token = e1_token.index_select(0, sort_idx)
        e2_token = e2_token.index_select(0, sort_idx)
        e1_length = e1_length.index_select(0, sort_idx)
        e2_length = e2_length.index_select(0, sort_idx)

        e1_type = e1_type.index_select(0, sort_idx)
        e2_type = e2_type.index_select(0, sort_idx)

        tok_num_betw = tok_num_betw.index_select(0, sort_idx)

        et_num = et_num.index_select(0, sort_idx)

        if y is not None:
            y = y.index_select(0, sort_idx)

        return [tokens, postag, positions1, positions2, e1_token, e2_token], \
               [e1_length, e2_length, e1_type, e2_type, tok_num_betw, et_num, sort_len, sort_idx], y
    else:
        return [tokens, postag, positions1, positions2, e1_token, e2_token], \
               [e1_length, e2_length, e1_type, e2_type, tok_num_betw, et_num, lengths, None], y


def pad_sequence(x, max_len, eos_idx):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    padded_x.fill(eos_idx)
    for i, row in enumerate(x):
        assert eos_idx not in row, 'EOS in sequence {}'.format(row)
        padded_x[i][:len(row)] = row
    padded_x = torch.LongTensor(padded_x)

    return padded_x

def endless_get_next_batch_without_rebatch(loaders, iters):
    try:
        x2, x1, y = next(iters)
    except StopIteration:
        iters = iter(loaders)
        x2, x1, y = next(iters)

    if len(y) < 2:
        return endless_get_next_batch_without_rebatch(loaders, iters)
    return x2, x1, y



def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True





