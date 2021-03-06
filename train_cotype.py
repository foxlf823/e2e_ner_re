from model.seqmodel import SeqModel, SeqModel1
from model.wordsequence import WordSequence
from classifymodel import ClassifyModel, ClassifyModel1
import torch
import itertools
import torch.optim as optim
import my_utils
from torch.utils.data import DataLoader
import time
import random
from ner import batchify_with_label
import sys
import ner
import os
import relation_extraction
from options import opt

def makeRelationDataset(re_X_positive, re_Y_positive, re_X_negative, re_Y_negative, ratio, b_shuffle, my_collate, batch_size):

    a = range(len(re_X_negative))
    random.shuffle(a)
    indices = a[:int(len(re_X_negative)*ratio)]

    temp_X = []
    temp_Y = []
    for i in range(len(re_X_positive)):
        temp_X.append(re_X_positive[i])
        temp_Y.append(re_Y_positive[i])
    for i in range(len(indices)):
        temp_X.append(re_X_negative[indices[i]])
        temp_Y.append(re_Y_negative[indices[i]])

    data_set = my_utils.RelationDataset(temp_X, temp_Y)

    data_loader = DataLoader(data_set, batch_size, shuffle=b_shuffle, collate_fn=my_collate)
    it = iter(data_loader)
    return data_loader, it

def pipeline(data, ner_dir, re_dir):

    seq_model = SeqModel(data)
    seq_wordseq = WordSequence(data, False, True, True, data.use_char)

    classify_wordseq = WordSequence(data, True, False, True, False)
    classify_model = ClassifyModel(data)
    if torch.cuda.is_available():
        classify_model = classify_model.cuda(data.HP_gpu)


    iter_parameter = itertools.chain(*map(list, [seq_wordseq.parameters(), seq_model.parameters()]))
    seq_optimizer = optim.Adam(iter_parameter, lr=opt.ner_lr, weight_decay=data.HP_l2)
    iter_parameter = itertools.chain(*map(list, [classify_wordseq.parameters(), classify_model.parameters()]))
    classify_optimizer = optim.Adam(iter_parameter, lr=opt.re_lr, weight_decay=data.HP_l2)



    if data.tune_wordemb == False:
        my_utils.freeze_net(seq_wordseq.wordrep.word_embedding)
        my_utils.freeze_net(classify_wordseq.wordrep.word_embedding)


    re_X_positive = []
    re_Y_positive = []
    re_X_negative = []
    re_Y_negative = []
    relation_vocab = data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']]
    my_collate = my_utils.sorted_collate1
    for i in range(len(data.re_train_X)):
        x = data.re_train_X[i]
        y = data.re_train_Y[i]

        if y != relation_vocab.get_index("</unk>"):
            re_X_positive.append(x)
            re_Y_positive.append(y)
        else:
            re_X_negative.append(x)
            re_Y_negative.append(y)

    re_test_loader = DataLoader(my_utils.RelationDataset(data.re_test_X, data.re_test_Y), data.HP_batch_size, shuffle=False, collate_fn=my_collate)

    best_ner_score = -1
    best_re_score = -1

    for idx in range(data.HP_iteration):
        epoch_start = time.time()

        seq_wordseq.train()
        seq_wordseq.zero_grad()
        seq_model.train()
        seq_model.zero_grad()

        classify_wordseq.train()
        classify_wordseq.zero_grad()
        classify_model.train()
        classify_model.zero_grad()

        batch_size = data.HP_batch_size

        random.shuffle(data.train_Ids)
        ner_train_num = len(data.train_Ids)
        ner_total_batch = ner_train_num // batch_size + 1

        re_train_loader, re_train_iter = makeRelationDataset(re_X_positive, re_Y_positive, re_X_negative, re_Y_negative,
                                                             data.unk_ratio, True, my_collate, data.HP_batch_size)
        re_total_batch = len(re_train_loader)

        total_batch = max(ner_total_batch, re_total_batch)
        min_batch = min(ner_total_batch, re_total_batch)

        for batch_id in range(total_batch):


            if batch_id < ner_total_batch:
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > ner_train_num:
                    end = ner_train_num
                instance = data.train_Ids[start:end]
                batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, \
                    batch_permute_label = batchify_with_label(instance, data.HP_gpu)

                hidden = seq_wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                             batch_charrecover, None, None)
                hidden_adv = None
                loss, tag_seq = seq_model.neg_log_likelihood_loss(hidden, hidden_adv, batch_label, mask)
                loss.backward()
                seq_optimizer.step()
                seq_wordseq.zero_grad()
                seq_model.zero_grad()


            if batch_id < re_total_batch:
                [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
                 batch_char, batch_charlen, batch_charrecover, \
                 position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
                 tok_num_betw, et_num], [targets, targets_permute] = my_utils.endless_get_next_batch_without_rebatch1(
                    re_train_loader, re_train_iter)

                hidden = classify_wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                                  batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                hidden_adv = None
                loss, pred = classify_model.neg_log_likelihood_loss(hidden, hidden_adv, batch_wordlen,
                                                                    e1_token, e1_length, e2_token, e2_length, e1_type,
                                                                    e2_type,
                                                                    tok_num_betw, et_num, targets)
                loss.backward()
                classify_optimizer.step()
                classify_wordseq.zero_grad()
                classify_model.zero_grad()


        epoch_finish = time.time()
        print("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))

        # _, _, _, _, f, _, _ = ner.evaluate(data, seq_wordseq, seq_model, "test")
        ner_score = ner.evaluate1(data, seq_wordseq, seq_model, "test")
        print("ner evaluate: f: %.4f" % (ner_score))

        re_score = relation_extraction.evaluate(classify_wordseq, classify_model, re_test_loader)
        print("re evaluate: f: %.4f" % (re_score))

        if ner_score+re_score > best_ner_score+best_re_score:
            print("new best score: ner: %.4f , re: %.4f" % (ner_score, re_score))
            best_ner_score = ner_score
            best_re_score = re_score

            torch.save(seq_wordseq.state_dict(), os.path.join(ner_dir, 'wordseq.pkl'))
            torch.save(seq_model.state_dict(), os.path.join(ner_dir, 'model.pkl'))
            torch.save(classify_wordseq.state_dict(), os.path.join(re_dir, 'wordseq.pkl'))
            torch.save(classify_model.state_dict(), os.path.join(re_dir, 'model.pkl'))


        # if ner_score > best_ner_score:
        #     print("new best ner score: %.4f" % (ner_score))
        #     best_ner_score = ner_score
        #
        #     torch.save(seq_wordseq.state_dict(), os.path.join(ner_dir, 'wordseq.pkl'))
        #     torch.save(seq_model.state_dict(), os.path.join(ner_dir, 'model.pkl'))
        #
        #
        # if re_score > best_re_score:
        #     print("new best re score: %.4f" % (re_score))
        #     best_re_score = re_score
        #
        #     torch.save(classify_wordseq.state_dict(), os.path.join(re_dir, 'wordseq.pkl'))
        #     torch.save(classify_model.state_dict(), os.path.join(re_dir, 'model.pkl'))


def hard(data, ner_dir, re_dir):

    seq_model = SeqModel1(data)
    seq_wordseq = WordSequence(data, False, True, True, data.use_char)

    classify_wordseq = WordSequence(data, True, False, True, False)
    classify_model = ClassifyModel1(data)

    wordseq_shared = WordSequence(data, False, False, False, False)


    iter_parameter = itertools.chain(*map(list, [seq_wordseq.parameters(), wordseq_shared.parameters(), seq_model.parameters()]))
    seq_optimizer = optim.Adam(iter_parameter, lr=opt.ner_lr, weight_decay=data.HP_l2)
    iter_parameter = itertools.chain(*map(list, [classify_wordseq.parameters(), wordseq_shared.parameters(), classify_model.parameters()]))
    classify_optimizer = optim.Adam(iter_parameter, lr=opt.re_lr, weight_decay=data.HP_l2)



    if data.tune_wordemb == False:
        my_utils.freeze_net(seq_wordseq.wordrep.word_embedding)
        my_utils.freeze_net(classify_wordseq.wordrep.word_embedding)
        my_utils.freeze_net(wordseq_shared.wordrep.word_embedding)

    re_X_positive = []
    re_Y_positive = []
    re_X_negative = []
    re_Y_negative = []
    relation_vocab = data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']]
    my_collate = my_utils.sorted_collate1
    for i in range(len(data.re_train_X)):
        x = data.re_train_X[i]
        y = data.re_train_Y[i]

        if y != relation_vocab.get_index("</unk>"):
            re_X_positive.append(x)
            re_Y_positive.append(y)
        else:
            re_X_negative.append(x)
            re_Y_negative.append(y)

    re_test_loader = DataLoader(my_utils.RelationDataset(data.re_test_X, data.re_test_Y), data.HP_batch_size, shuffle=False, collate_fn=my_collate)

    best_ner_score = -1
    best_re_score = -1

    for idx in range(data.HP_iteration):
        epoch_start = time.time()

        seq_wordseq.train()
        seq_wordseq.zero_grad()
        seq_model.train()
        seq_model.zero_grad()

        classify_wordseq.train()
        classify_wordseq.zero_grad()
        classify_model.train()
        classify_model.zero_grad()

        wordseq_shared.train()
        wordseq_shared.zero_grad()

        batch_size = data.HP_batch_size

        random.shuffle(data.train_Ids)
        ner_train_num = len(data.train_Ids)
        ner_total_batch = ner_train_num // batch_size + 1

        re_train_loader, re_train_iter = makeRelationDataset(re_X_positive, re_Y_positive, re_X_negative, re_Y_negative,
                                                             data.unk_ratio, True, my_collate, data.HP_batch_size)
        re_total_batch = len(re_train_loader)

        total_batch = max(ner_total_batch, re_total_batch)
        # min_batch = min(ner_total_batch, re_total_batch)

        for batch_id in range(total_batch):


            if batch_id < ner_total_batch:
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > ner_train_num:
                    end = ner_train_num
                instance = data.train_Ids[start:end]
                batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, \
                    batch_permute_label = batchify_with_label(instance, data.HP_gpu)

                # if batch_id < min_batch:
                hidden = seq_wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, None, None)
                hidden_shared = wordseq_shared.forward(batch_word, None, batch_wordlen, None, None, None, None, None)
                loss, tag_seq = seq_model.neg_log_likelihood_loss(hidden, hidden_shared, batch_label, mask)
                loss.backward()
                seq_optimizer.step()
                seq_wordseq.zero_grad()
                wordseq_shared.zero_grad()
                seq_model.zero_grad()

            if batch_id < re_total_batch:
                [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
                 batch_char, batch_charlen, batch_charrecover, \
                 position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
                 tok_num_betw, et_num], [targets, targets_permute] = my_utils.endless_get_next_batch_without_rebatch1(
                    re_train_loader, re_train_iter)

                # if batch_id < min_batch:
                hidden = classify_wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                         batch_charrecover, position1_seq_tensor, position2_seq_tensor)
                hidden_shared = wordseq_shared.forward(batch_word, None, batch_wordlen, None, None, None, None, None)
                loss, pred = classify_model.neg_log_likelihood_loss(hidden, hidden_shared, batch_wordlen,
                                                           e1_token, e1_length, e2_token, e2_length, e1_type,
                                                           e2_type,
                                                           tok_num_betw, et_num, targets)
                loss.backward()
                classify_optimizer.step()
                classify_wordseq.zero_grad()
                wordseq_shared.zero_grad()
                classify_model.zero_grad()


        epoch_finish = time.time()
        print("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))

        # _, _, _, _, f, _, _ = ner.evaluate(data, seq_wordseq, seq_model, "test")
        ner_score = ner.evaluate2(data, seq_wordseq, wordseq_shared, seq_model, "test")
        print("ner evaluate: f: %.4f" % (ner_score))

        re_score = relation_extraction.evaluate1(classify_wordseq, wordseq_shared, classify_model, re_test_loader)
        print("re evaluate: f: %.4f" % (re_score))

        if ner_score+re_score > best_ner_score+best_re_score:
            print("new best score: ner: %.4f , re: %.4f" % (ner_score, re_score))
            best_ner_score = ner_score
            best_re_score = re_score

            torch.save(seq_wordseq.state_dict(), os.path.join(ner_dir, 'wordseq.pkl'))
            torch.save(seq_model.state_dict(), os.path.join(ner_dir, 'model.pkl'))
            torch.save(classify_wordseq.state_dict(), os.path.join(re_dir, 'wordseq.pkl'))
            torch.save(classify_model.state_dict(), os.path.join(re_dir, 'model.pkl'))
            torch.save(wordseq_shared.state_dict(), os.path.join(ner_dir, 'wordseq_shared.pkl'))

