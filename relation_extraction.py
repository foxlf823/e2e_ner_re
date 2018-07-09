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


import my_utils1
from feature_extractor import *
# from utils.data import data
from data_structure import *
import utils.functions
from classifymodel import ClassifyModel

# def dataset_stat(tokens, entities, relations):
#     word_alphabet = sortedcontainers.SortedSet()
#     postag_alphabet = sortedcontainers.SortedSet()
#     relation_alphabet = sortedcontainers.SortedSet()
#     entity_type_alphabet = sortedcontainers.SortedSet()
#     entity_alphabet = sortedcontainers.SortedSet()
#
#     for i, doc_token in enumerate(tokens):
#
#         doc_entity = entities[i]
#         doc_relation = relations[i]
#
#         sent_idx = 0
#         sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]
#         while sentence.shape[0] != 0:
#             for _, token in sentence.iterrows():
#                 word_alphabet.add(my_utils.normalizeWord(token['text']))
#                 postag_alphabet.add(token['postag'])
#
#             entities_in_sentence = doc_entity[(doc_entity['sent_idx'] == sent_idx)]
#             for _, entity in entities_in_sentence.iterrows():
#                 entity_type_alphabet.add(entity['type'])
#                 tk_idx = entity['tf_start']
#                 while tk_idx <= entity['tf_end']:
#                     entity_alphabet.add(my_utils.normalizeWord(sentence.iloc[tk_idx, 0])) # assume 'text' is in 0 column
#                     tk_idx += 1
#
#             sent_idx += 1
#             sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]
#
#         for _, relation in doc_relation.iterrows():
#             relation_alphabet.add(relation['type'])
#
#     return word_alphabet, postag_alphabet, relation_alphabet, entity_type_alphabet, entity_alphabet
#



# def pretrain(train_token, train_entity, train_relation, train_name, test_token, test_entity, test_relation, test_name,
#              data):
#     word_alphabet, postag_alphabet, relation_alphabet, entity_type_alphabet, entity_alphabet = dataset_stat(train_token, train_entity, train_relation)
#     logging.info("training dataset stat completed")
#     if data.full_data:
#         test_word_alphabet, test_postag_alphabet, test_relation_alphabet, test_entity_type_alphabet, test_entity_alphabet = dataset_stat(test_token, test_entity, test_relation)
#         word_alphabet = word_alphabet | test_word_alphabet
#         postag_alphabet = postag_alphabet | test_postag_alphabet
#         relation_alphabet = relation_alphabet | test_relation_alphabet
#         entity_type_alphabet = entity_type_alphabet | test_entity_type_alphabet
#         entity_alphabet = entity_alphabet | test_entity_alphabet
#         del test_word_alphabet, test_postag_alphabet, test_relation_alphabet, test_entity_type_alphabet, test_entity_alphabet
#         logging.info("test dataset stat completed")
#
#     position_alphabet = sortedcontainers.SortedSet()
#     for i in range(data.max_seq_len):
#         position_alphabet.add(i)
#         position_alphabet.add(-i)
#
#     relation_vocab = vocab.Vocab(relation_alphabet, None, data.feat_config['[RELATION]']['emb_size'], data, data.feat_config['[RELATION]']['emb_norm'])
#     word_vocab = vocab.Vocab(word_alphabet, data.word_emb_dir, data.word_emb_dim, data, data.norm_word_emb)
#     postag_vocab = vocab.Vocab(postag_alphabet, None, data.feat_config['[POS]']['emb_size'], data, data.feat_config['[POS]']['emb_norm'])
#     entity_type_vocab = vocab.Vocab(entity_type_alphabet, None, data.feat_config['[ENTITY_TYPE]']['emb_size'], data, data.feat_config['[ENTITY_TYPE]']['emb_norm'])
#     entity_vocab = vocab.Vocab(entity_alphabet, None, data.feat_config['[ENTITY]']['emb_size'], data, data.feat_config['[ENTITY]']['emb_norm'])
#     position_vocab1 = vocab.Vocab(position_alphabet, None, data.feat_config['[POSITION]']['emb_size'], data, data.feat_config['[POSITION]']['emb_norm'])
#     position_vocab2 = vocab.Vocab(position_alphabet, None, data.feat_config['[POSITION]']['emb_size'], data, data.feat_config['[POSITION]']['emb_norm'])
#     # we directly use position_alphabet to build them, since they are all numbers
#     tok_num_betw_vocab = vocab.Vocab(position_alphabet, None, data.feat_config['[POSITION]']['emb_size'], data, data.feat_config['[POSITION]']['emb_norm'])
#     et_num_vocab = vocab.Vocab(position_alphabet, None, data.feat_config['[POSITION]']['emb_size'], data, data.feat_config['[POSITION]']['emb_norm'])
#     logging.info("vocab build completed")
#
#     logging.info("saving ... vocab")
#     pickle.dump(word_vocab, open(os.path.join(data.pretrain, 'word_vocab.pkl'), "wb"), True)
#     pickle.dump(postag_vocab, open(os.path.join(data.pretrain, 'postag_vocab.pkl'), "wb"), True)
#     pickle.dump(relation_vocab, open(os.path.join(data.pretrain, 'relation_vocab.pkl'), "wb"), True)
#     pickle.dump(entity_type_vocab, open(os.path.join(data.pretrain, 'entity_type_vocab.pkl'), "wb"), True)
#     pickle.dump(entity_vocab, open(os.path.join(data.pretrain, 'entity_vocab.pkl'), "wb"), True)
#     pickle.dump(position_vocab1, open(os.path.join(data.pretrain, 'position_vocab1.pkl'), "wb"), True)
#     pickle.dump(position_vocab2, open(os.path.join(data.pretrain, 'position_vocab2.pkl'), "wb"), True)
#     pickle.dump(tok_num_betw_vocab, open(os.path.join(data.pretrain, 'tok_num_betw_vocab.pkl'), "wb"), True)
#     pickle.dump(et_num_vocab, open(os.path.join(data.pretrain, 'et_num_vocab.pkl'), "wb"), True)
#
#     train_X, train_Y, _ = my_utils.getRelationInstance2(train_token, train_entity, train_relation, train_name, word_vocab, postag_vocab,
#                                                      relation_vocab, entity_type_vocab,
#                                                      entity_vocab, position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab)
#     logging.info("training instance build completed, total {}".format(len(train_Y)))
#     pickle.dump(train_X, open(os.path.join(data.pretrain, 'train_X.pkl'), "wb"), True)
#     pickle.dump(train_Y, open(os.path.join(data.pretrain, 'train_Y.pkl'), "wb"), True)
#
#
#     test_X, test_Y, test_other = my_utils.getRelationInstance2(test_token, test_entity, test_relation, test_name, word_vocab, postag_vocab,
#                                                             relation_vocab, entity_type_vocab,
#                                                             entity_vocab, position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab)
#     logging.info("test instance build completed, total {}".format(len(test_Y)))
#     pickle.dump(test_X, open(os.path.join(data.pretrain, 'test_X.pkl'), "wb"), True)
#     pickle.dump(test_Y, open(os.path.join(data.pretrain, 'test_Y.pkl'), "wb"), True)
#     pickle.dump(test_other, open(os.path.join(data.pretrain, 'test_Other.pkl'), "wb"), True)


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

def train(data, dir):

    # cnnrnn
    if data.feature_extractor == 'lstm':
        my_collate = my_utils.sorted_collate
    else:
        my_collate = my_utils.unsorted_collate


    train_loader, train_iter = makeDatasetWithoutUnknown(data.re_train_X, data.re_train_Y, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']], True, my_collate, data.HP_batch_size)
    num_iter = len(train_loader)
    unk_loader, unk_iter = makeDatasetUnknown(data.re_train_X, data.re_train_Y, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']], my_collate, data.unk_ratio, data.HP_batch_size)

    test_loader = DataLoader(my_utils.RelationDataset(data.re_test_X, data.re_test_Y),
                              data.HP_batch_size, shuffle=False, collate_fn=my_collate)


    # cnnrnn
    if data.feature_extractor == 'lstm':
        m_low = LSTMFeatureExtractor(data, 1, data.seq_feature_size, data.HP_dropout, data.HP_gpu)
    if torch.cuda.is_available():
        m_low = m_low.cuda(data.HP_gpu)


    m = MLP(data.seq_feature_size, data)
    if torch.cuda.is_available():
        m = m.cuda(data.HP_gpu)

    iter_parameter = itertools.chain(*map(list, [m_low.parameters(), m.parameters()]))
    optimizer = optim.Adam(iter_parameter, lr=data.HP_lr)

    if data.tune_wordemb == False:
        my_utils.freeze_net(m_low.word_emb)

    best_acc = 0.0
    logging.info("start training ...")
    for epoch in range(data.max_epoch):
        m_low.train()
        m.train()
        correct, total = 0, 0

        for i in tqdm(range(num_iter)):

            x2, x1, targets = my_utils.endless_get_next_batch_without_rebatch(train_loader, train_iter)

            hidden_features = m_low.forward(x2, x1)

            outputs = m.forward(hidden_features, x2, x1)
            loss = m.loss(targets, outputs)
            # logging.info("output: {}".format(outputs))
            # logging.info("loss: {}".format(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += targets.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == targets).sum().item()


            x2, x1, targets = my_utils.endless_get_next_batch_without_rebatch(unk_loader, unk_iter)

            hidden_features = m_low.forward(x2, x1)

            outputs = m.forward(hidden_features, x2, x1)
            loss = m.loss(targets, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        unk_loader, unk_iter = makeDatasetUnknown(data.re_train_X, data.re_train_Y,
                                                  data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']],
                                                  my_collate, data.unk_ratio, data.HP_batch_size)

        logging.info('epoch {} end'.format(epoch))
        logging.info('Train Accuracy: {}%'.format(100.0 * correct / total))

        test_accuracy = evaluate(m_low, m, test_loader, None)
        # test_accuracy = evaluate(m, test_loader)
        logging.info('Test Accuracy: {}%'.format(test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(m_low.state_dict(), '{}/feature_extractor.pth'.format(dir))
            torch.save(m.state_dict(), '{}/model.pth'.format(dir))
            logging.info('New best accuracy: {}'.format(best_acc))


    logging.info("training completed")


def train1(data, dir):

    my_collate = my_utils.sorted_collate1

    train_loader, train_iter = makeDatasetWithoutUnknown(data.re_train_X, data.re_train_Y, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']], True, my_collate, data.HP_batch_size)
    num_iter = len(train_loader)
    unk_loader, unk_iter = makeDatasetUnknown(data.re_train_X, data.re_train_Y, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']], my_collate, data.unk_ratio, data.HP_batch_size)

    test_loader = DataLoader(my_utils.RelationDataset(data.re_test_X, data.re_test_Y),
                              data.HP_batch_size, shuffle=False, collate_fn=my_collate)


    model = ClassifyModel(data)
    if torch.cuda.is_available():
        model = model.cuda(data.HP_gpu)

    optimizer = optim.Adam(model.parameters(), lr=data.HP_lr)

    if data.tune_wordemb == False:
        my_utils.freeze_net(model.word_hidden.wordrep.word_embedding)

    best_acc = 0.0
    logging.info("start training ...")
    for epoch in range(data.max_epoch):

        model.train()
        correct, total = 0, 0

        for i in tqdm(range(num_iter)):
            [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, \
            position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
            tok_num_betw, et_num], targets = my_utils.endless_get_next_batch_without_rebatch1(train_loader, train_iter)


            loss, pred = model.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                                                       e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num, targets)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += targets.size(0)
            correct += (pred == targets).sum().item()


            [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, \
            position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
            tok_num_betw, et_num], targets = my_utils.endless_get_next_batch_without_rebatch1(unk_loader, unk_iter)

            loss, pred = model.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                                                       e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num, targets)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        unk_loader, unk_iter = makeDatasetUnknown(data.re_train_X, data.re_train_Y,
                                                  data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']],
                                                  my_collate, data.unk_ratio, data.HP_batch_size)

        logging.info('epoch {} end'.format(epoch))
        logging.info('Train Accuracy: {}%'.format(100.0 * correct / total))

        test_accuracy = evaluate1(model, test_loader)
        # test_accuracy = evaluate(m, test_loader)
        logging.info('Test Accuracy: {}%'.format(test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), '{}/model.pkl'.format(dir))
            logging.info('New best accuracy: {}'.format(best_acc))


    logging.info("training completed")


def evaluate(feature_extractor, m, loader, other):
    #results = []
    feature_extractor.eval()
    m.eval()
    it = iter(loader)
    # start, end = 0, 0
    correct = 0
    total = 0
    # iii = 0
    for x2, x1, targets in it:


        with torch.no_grad():

            _, _, _, _, _, _, _, sort_idx = x1

            hidden_features = feature_extractor.forward(x2, x1)


            outputs = m.forward(hidden_features, x2, x1)


            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().data.item()

        # start = end
        # end = end + targets.size(0)

        # we use sorted_collate, so we need to unsorted them during evaluate
        # cnnrnn
        # _, unsort_idx = sort_idx.sort(0, descending=False)
        # pred = pred.index_select(0, unsort_idx)
        #
        # for i, d in enumerate(other[start:end]):
        #     d["type"] = pred[i].item()

        # iii += 1

    acc = 100.0 * correct / total
    return acc

def evaluate1(model, loader):

    model.eval()
    it = iter(loader)
    correct = 0
    total = 0

    for [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, \
            position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
            tok_num_betw, et_num], targets in it:


        with torch.no_grad():

            pred = model.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                                                       e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num)

            total += targets.size(0)
            correct += (pred == targets).sum().data.item()

    acc = 100.0 * correct / total
    return acc

def evaluateWhenTest(feature_extractor, m, instances, data, test_other, relationVocab):

    feature_extractor.eval()
    m.eval()
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

        # cnnrnn
        if data.feature_extractor == 'lstm':
            x2, x1, _ = my_utils.my_collate(instance, True)
        else:
            x2, x1, _ = my_utils.my_collate(instance, False)

        with torch.no_grad():

            _, _, _, _, _, _, _, sort_idx = x1

            hidden_features = feature_extractor.forward(x2, x1)

            outputs = m.forward(hidden_features, x2, x1)

            _, pred = torch.max(outputs, 1)


        # we use sorted_collate, so we need to unsorted them during evaluate
        # cnnrnn
        if data.feature_extractor == 'lstm':
            _, unsort_idx = sort_idx.sort(0, descending=False)
            pred = pred.index_select(0, unsort_idx)



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

def evaluateWhenTest1(model, instances, data, test_other, relationVocab):

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
         tok_num_betw, et_num], targets = my_utils.sorted_collate1(instance)

        with torch.no_grad():

            pred = model.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
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
