import logging
import os
import cPickle as pickle
import sortedcontainers
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import random
import torch
import itertools
import bioc
import math
import pandas as pd
from options import opt


import my_utils1
from feature_extractor import *
# from utils.data import data
from data_structure import *
import utils.functions
from classifymodel import ClassifyModel
from model.wordsequence import WordSequence


def makeDatasetWithoutUnknown(test_X, test_Y, relation_vocab, b_shuffle, my_collate, batch_size):
    test_X_remove_unk = []
    test_Y_remove_unk = []
    for i in range(len(test_X)):
        x = test_X[i]
        y = test_Y[i]

        if y != relation_vocab.get_index("</unk>"):
            test_X_remove_unk.append(x)
            test_Y_remove_unk.append(y)

    test_set = my_utils.RelationDataset(test_X_remove_unk, test_Y_remove_unk)
    test_loader = DataLoader(test_set, batch_size, shuffle=b_shuffle, collate_fn=my_collate)
    it = iter(test_loader)
    logging.info("instance after removing unknown, {}".format(len(test_Y_remove_unk)))
    return test_loader, it

def randomSampler(dataset_list, ratio):
    a = range(len(dataset_list))
    random.shuffle(a)
    indices = a[:int(len(dataset_list)*ratio)]
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    return sampler

def makeDatasetUnknown(test_X, test_Y, relation_vocab, my_collate, ratio, batch_size):
    test_X_remove_unk = []
    test_Y_remove_unk = []
    for i in range(len(test_X)):
        x = test_X[i]
        y = test_Y[i]

        if y == relation_vocab.get_index("</unk>"):
            test_X_remove_unk.append(x)
            test_Y_remove_unk.append(y)

    test_set = my_utils.RelationDataset(test_X_remove_unk, test_Y_remove_unk)

    test_loader = DataLoader(test_set, batch_size, shuffle=False, sampler=randomSampler(test_Y_remove_unk, ratio), collate_fn=my_collate)
    it = iter(test_loader)

    return test_loader, it


def train1(data, dir):

    my_collate = my_utils.sorted_collate1

    train_loader, train_iter = makeDatasetWithoutUnknown(data.re_train_X, data.re_train_Y, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']], True, my_collate, data.HP_batch_size)
    num_iter = len(train_loader)
    unk_loader, unk_iter = makeDatasetUnknown(data.re_train_X, data.re_train_Y, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']], my_collate, data.unk_ratio, data.HP_batch_size)

    test_loader = DataLoader(my_utils.RelationDataset(data.re_test_X, data.re_test_Y),
                              data.HP_batch_size, shuffle=False, collate_fn=my_collate)

    wordseq = WordSequence(data, True, False, False)
    model = ClassifyModel(data)
    if torch.cuda.is_available():
        model = model.cuda(data.HP_gpu)

    if opt.self_adv == 'grad':
        wordseq_adv = WordSequence(data, True, False, False)
    elif opt.self_adv == 'label':
        wordseq_adv = WordSequence(data, True, False, False)
        model_adv = ClassifyModel(data)
        if torch.cuda.is_available():
            model_adv = model_adv.cuda(data.HP_gpu)
    else:
        wordseq_adv = None

    if opt.self_adv == 'grad':
        iter_parameter = itertools.chain(
            *map(list, [wordseq.parameters(), wordseq_adv.parameters(), model.parameters()]))
        optimizer = optim.Adam(iter_parameter, lr=data.HP_lr, weight_decay=data.HP_l2)
    elif opt.self_adv == 'label':
        iter_parameter = itertools.chain(*map(list, [wordseq.parameters(), model.parameters()]))
        optimizer = optim.Adam(iter_parameter, lr=data.HP_lr, weight_decay=data.HP_l2)
        iter_parameter = itertools.chain(*map(list, [wordseq_adv.parameters(), model_adv.parameters()]))
        optimizer_adv = optim.Adam(iter_parameter, lr=data.HP_lr, weight_decay=data.HP_l2)

    else:
        iter_parameter = itertools.chain(*map(list, [wordseq.parameters(), model.parameters()]))
        optimizer = optim.Adam(iter_parameter, lr=data.HP_lr, weight_decay=data.HP_l2)

    if data.tune_wordemb == False:
        my_utils.freeze_net(wordseq.wordrep.word_embedding)
        if opt.self_adv != 'no':
            my_utils.freeze_net(wordseq_adv.wordrep.word_embedding)

    best_acc = 0.0
    logging.info("start training ...")
    for epoch in range(data.max_epoch):
        wordseq.train()
        wordseq.zero_grad()
        model.train()
        model.zero_grad()
        if opt.self_adv == 'grad':
            wordseq_adv.train()
            wordseq_adv.zero_grad()
        elif opt.self_adv == 'label':
            wordseq_adv.train()
            wordseq_adv.zero_grad()
            model_adv.train()
            model_adv.zero_grad()
        correct, total = 0, 0

        for i in range(num_iter):
            [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, \
            position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
            tok_num_betw, et_num], [targets, targets_permute] = my_utils.endless_get_next_batch_without_rebatch1(train_loader, train_iter)


            if opt.self_adv == 'grad':
                hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                hidden_adv = wordseq_adv.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                loss, pred = model.neg_log_likelihood_loss(hidden, hidden_adv, batch_wordlen,
                                                           e1_token, e1_length, e2_token, e2_length, e1_type, e2_type,
                                                           tok_num_betw, et_num, targets)
                loss.backward()
                my_utils.reverse_grad(wordseq_adv)
                optimizer.step()
                wordseq.zero_grad()
                wordseq_adv.zero_grad()
                model.zero_grad()

            elif opt.self_adv == 'label' :
                wordseq.unfreeze_net()
                wordseq_adv.freeze_net()
                hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                hidden_adv = wordseq_adv.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                loss, pred = model.neg_log_likelihood_loss(hidden, hidden_adv, batch_wordlen,
                                                           e1_token, e1_length, e2_token, e2_length, e1_type, e2_type,
                                                           tok_num_betw, et_num, targets)
                loss.backward()
                optimizer.step()
                wordseq.zero_grad()
                wordseq_adv.zero_grad()
                model.zero_grad()

                wordseq.freeze_net()
                wordseq_adv.unfreeze_net()
                hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                hidden_adv = wordseq_adv.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                loss_adv, _ = model_adv.neg_log_likelihood_loss(hidden, hidden_adv, batch_wordlen,
                                                           e1_token, e1_length, e2_token, e2_length, e1_type, e2_type,
                                                           tok_num_betw, et_num, targets_permute)
                loss_adv.backward()
                optimizer_adv.step()
                wordseq.zero_grad()
                wordseq_adv.zero_grad()
                model_adv.zero_grad()

            else:
                hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                hidden_adv = None
                loss, pred = model.neg_log_likelihood_loss(hidden, hidden_adv, batch_wordlen,
                                                           e1_token, e1_length, e2_token, e2_length, e1_type, e2_type,
                                                           tok_num_betw, et_num, targets)
                loss.backward()
                optimizer.step()
                wordseq.zero_grad()
                model.zero_grad()



            total += targets.size(0)
            correct += (pred == targets).sum().item()


            [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, \
            position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
            tok_num_betw, et_num], [targets, targets_permute] = my_utils.endless_get_next_batch_without_rebatch1(unk_loader, unk_iter)

            hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                     batch_charrecover, position1_seq_tensor, position2_seq_tensor)
            hidden_adv = None
            loss, pred = model.neg_log_likelihood_loss(hidden, hidden_adv, batch_wordlen,
                                                       e1_token, e1_length, e2_token, e2_length, e1_type, e2_type,
                                                       tok_num_betw, et_num, targets)
            loss.backward()
            optimizer.step()
            wordseq.zero_grad()
            model.zero_grad()


        unk_loader, unk_iter = makeDatasetUnknown(data.re_train_X, data.re_train_Y,
                                                  data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']],
                                                  my_collate, data.unk_ratio, data.HP_batch_size)

        logging.info('epoch {} end'.format(epoch))
        logging.info('Train Accuracy: {}%'.format(100.0 * correct / total))

        test_accuracy = evaluate1(wordseq, model, test_loader)
        # test_accuracy = evaluate(m, test_loader)
        logging.info('Test Accuracy: {}%'.format(test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(wordseq.state_dict(), os.path.join(dir, 'wordseq.pkl'))
            torch.save(model.state_dict(), '{}/model.pkl'.format(dir))
            if opt.self_adv == 'grad':
                torch.save(wordseq_adv.state_dict(), os.path.join(dir, 'wordseq_adv.pkl'))
            elif opt.self_adv == 'label':
                torch.save(wordseq_adv.state_dict(), os.path.join(dir, 'wordseq_adv.pkl'))
                torch.save(model_adv.state_dict(), os.path.join(dir, 'model_adv.pkl'))
            logging.info('New best accuracy: {}'.format(best_acc))


    logging.info("training completed")



def evaluate1(wordseq, model, loader):
    wordseq.eval()
    model.eval()
    it = iter(loader)
    correct = 0
    total = 0

    for [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, \
            position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
            tok_num_betw, et_num], [targets, targets_permute] in it:


        with torch.no_grad():
            hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                     batch_charrecover, position1_seq_tensor, position2_seq_tensor)
            pred = model.forward(hidden, batch_wordlen, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num)

            total += targets.size(0)
            correct += (pred == targets).sum().data.item()

    # acc = 100.0 * correct / total
    acc = 1.0 * correct / total
    return acc


def evaluateWhenTest1(wordseq, model, instances, data, test_other, relationVocab):
    wordseq.eval()
    model.eval()
    batch_size = data.HP_batch_size

    relations = []
    relation_id = 1

    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue

        [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
         batch_char, batch_charlen, batch_charrecover, \
         position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
         tok_num_betw, et_num], [targets, targets_permute] = my_utils.sorted_collate1(instance)

        with torch.no_grad():
            hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                     batch_charrecover, position1_seq_tensor, position2_seq_tensor)
            pred = model.forward(hidden, batch_wordlen,
                                 e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num)

            pred = pred.index_select(0, batch_wordrecover)


        for i in range(start,end):

            former = test_other[i][0]
            latter = test_other[i][1]

            relation_type = relationVocab.get_instance(pred[i-start].item())
            if relation_type == '</unk>':
                continue
            elif relationConstraint1(relation_type, former.type, latter.type) == False:
                continue
            else:
                relation = Relation()
                relation.create(str(relation_id), relation_type, former, latter)
                relations.append(relation)

                relation_id += 1

    return relations

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
    else:
        return 0


def getRelationInstance2(tokens, entities, relations, names, data):
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
                    cap = []
                    chars = []
                    positions1 = []
                    positions2 = []
                    former_token = []
                    latter_token = []
                    i = 0
                    for _, token in context_token.iterrows():
                        if data.number_normalized:
                            word = utils.functions.normalize_word(token['text'])
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
                            pass
                        else:
                            positions1.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(0))
                            former_token.append(data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(entity_word))

                        if i < latter_tf_start:
                            positions2.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(latter_tf_start - i))
                            pass
                        elif i > latter_tf_end:
                            positions2.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(latter_tf_end - i))
                            pass
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
                    if type_constraint == 1:
                        features['e1_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(former['type'])
                        features['e2_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(latter['type'])
                        features['e1_token'] = former_token
                        features['e2_token'] = latter_token
                    else:
                        features['e1_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(latter['type'])
                        features['e2_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(former['type'])
                        features['e1_token'] = latter_token
                        features['e2_token'] = former_token

                    features['tok_num_betw'] = data.re_feature_alphabets[data.re_feature_name2id['[TOKEN_NUM]']].get_index(latter['tf_start']-former['tf_end'])

                    entity_between = doc_entity[((doc_entity['start']>=former['end']) & (doc_entity['end']<=latter['start']))]
                    features['et_num'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_NUM]']].get_index(entity_between.shape[0])

                    X.append(features)

                    if gold_relations.shape[0] == 0:
                        Y.append(data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']].get_index('</unk>'))
                        cnt_neg += 1
                    else:
                        gold_answer = gold_relations.iloc[0]['type']
                        Y.append(data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']].get_index(gold_answer))


    neg = 100.0*cnt_neg/len(Y)

    logging.info("positive instance {}%, negative instance {}%".format(100-neg, neg))
    return X, Y


def getRelationInstanceForOneDoc(doc_token, entities, doc_name, data):
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
                cap = []
                chars = []
                positions1 = []
                positions2 = []
                former_token = []
                latter_token = []
                i = 0
                for _, token in context_token.iterrows():
                    if data.number_normalized:
                        word = utils.functions.normalize_word(token['text'])
                    else:
                        word = token['text']
                    entity_word = my_utils1.normalizeWord(token['text'])
                    words.append(data.word_alphabet.get_index(word))
                    postags.append(data.feature_alphabets[data.feature_name2id['[POS]']].get_index(token['postag']))
                    cap.append(data.feature_alphabets[data.feature_name2id['[Cap]']].get_index(
                        str(my_utils.featureCapital(token['text']))))
                    char_for1word = []
                    for char in word:
                        char_for1word.append(data.char_alphabet.get_index(char))
                    chars.append(char_for1word)

                    if i < former_tf_start:
                        positions1.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(
                            former_tf_start - i))

                    elif i > former_tf_end:
                        positions1.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(
                            former_tf_end - i))
                        pass
                    else:
                        positions1.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(0))
                        former_token.append(
                            data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(entity_word))

                    if i < latter_tf_start:
                        positions2.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(
                            latter_tf_start - i))
                        pass
                    elif i > latter_tf_end:
                        positions2.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(
                            latter_tf_end - i))
                        pass
                    else:
                        positions2.append(data.re_feature_alphabets[data.re_feature_name2id['[POSITION]']].get_index(0))
                        latter_token.append(
                            data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(entity_word))

                    i += 1

                if len(former_token) == 0: # truncated part contains entity, so we have to use the text in doc_entity
                    # splitted = re.split(r"\s+| +|[\(\)\[\]\-_,]+", former['text'])
                    splitted = my_utils.my_tokenize(former.text)
                    for s in splitted:
                        s = s.strip()
                        if s != "":
                            former_token.append(data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(my_utils1.normalizeWord(s)))
                if len(latter_token) == 0:
                    #splitted = re.split(r"\s+| +|[\(\)\[\]\-_,]+", latter['text'])
                    splitted = my_utils.my_tokenize(latter.text)
                    for s in splitted:
                        s = s.strip()
                        if s != "":
                            latter_token.append(data.re_feature_alphabets[data.re_feature_name2id['[ENTITY]']].get_index(my_utils1.normalizeWord(s)))

                assert len(former_token)>0
                assert len(latter_token)>0


                features = {'tokens': words, 'postag': postags, 'cap': cap, 'char': chars, 'positions1': positions1, 'positions2': positions2}
                if type_constraint == 1:
                    features['e1_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(former.type)
                    features['e2_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(latter.type)
                    features['e1_token'] = former_token
                    features['e2_token'] = latter_token
                else:
                    features['e1_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(latter.type)
                    features['e2_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(former.type)
                    features['e1_token'] = latter_token
                    features['e2_token'] = former_token

                features['tok_num_betw'] = data.re_feature_alphabets[data.re_feature_name2id['[TOKEN_NUM]']].get_index(latter.tf_start-former.tf_end)

                entity_between = getEntitiesBetween(former, latter, entities)
                features['et_num'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_NUM]']].get_index(len(entity_between))

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
