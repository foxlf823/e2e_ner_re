import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.wordsequence import WordSequence
from model.crf import CRF
from classifymodel import DotAttentionLayer
import my_utils
import itertools
from options import opt
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import random
from joint_train import makeRelationDataset
from ner import batchify_with_label
import ner
import relation_extraction
import os
import preprocess
import bioc
from tqdm import tqdm
from data_structure import *
from model.charcnn import CharCNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import preprocess_cotype
import test_cotype

class WordRep(nn.Module):
    def __init__(self, data, use_position, use_cap, use_postag, use_char):
        super(WordRep, self).__init__()

        self.gpu = data.HP_gpu
        self.use_char = use_char
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.char_all_feature = False
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim

            self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)

        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        self.feature_num = 0
        self.feature_embedding_dims = data.feature_emb_dims
        self.feature_embeddings = nn.ModuleList()

        if use_cap:
            self.feature_num += 1
            alphabet_id = data.feature_name2id['[Cap]']
            emb = nn.Embedding(data.feature_alphabets[alphabet_id].size(), self.feature_embedding_dims[alphabet_id])
            emb.weight.data.copy_(torch.from_numpy(
                self.random_embedding(data.feature_alphabets[alphabet_id].size(), self.feature_embedding_dims[alphabet_id])))
            self.feature_embeddings.append(emb)

        if use_postag:
            self.feature_num += 1
            alphabet_id = data.feature_name2id['[POS]']
            emb = nn.Embedding(data.feature_alphabets[alphabet_id].size(), self.feature_embedding_dims[alphabet_id])
            emb.weight.data.copy_(torch.from_numpy(
                self.random_embedding(data.feature_alphabets[alphabet_id].size(), self.feature_embedding_dims[alphabet_id])))
            self.feature_embeddings.append(emb)

        self.use_position = use_position
        if self.use_position:

            position_alphabet_id = data.re_feature_name2id['[POSITION]']
            self.position_embedding_dim = data.re_feature_emb_dims[position_alphabet_id]
            self.position1_emb = nn.Embedding(data.re_feature_alphabet_sizes[position_alphabet_id],
                                              self.position_embedding_dim, data.pad_idx)
            self.position1_emb.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.re_feature_alphabet_sizes[position_alphabet_id],
                                              self.position_embedding_dim)))

            self.position2_emb = nn.Embedding(data.re_feature_alphabet_sizes[position_alphabet_id],
                                              self.position_embedding_dim, data.pad_idx)
            self.position2_emb.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.re_feature_alphabet_sizes[position_alphabet_id],
                                              self.position_embedding_dim)))

        if torch.cuda.is_available():
            self.drop = self.drop.cuda(self.gpu)
            self.word_embedding = self.word_embedding.cuda(self.gpu)
            for idx in range(self.feature_num):
                self.feature_embeddings[idx] = self.feature_embeddings[idx].cuda(self.gpu)
            if self.use_position:
                self.position1_emb = self.position1_emb.cuda(self.gpu)
                self.position2_emb = self.position2_emb.cuda(self.gpu)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def forward(self, word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, position1_inputs,
                position2_inputs):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embedding(word_inputs)
        word_list = [word_embs]
        for idx in range(self.feature_num):
            word_list.append(self.feature_embeddings[idx](feature_inputs[idx]))

        if self.use_char:
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            ## concat word and char together
            word_list.append(char_features)
            word_embs = torch.cat([word_embs, char_features], 2)
            if self.char_all_feature:
                char_features_extra = self.char_feature_extra.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
                char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size,sent_len,-1)
                ## concat word and char together
                word_list.append(char_features_extra)

        if self.use_position:
            position1_feature = self.position1_emb(position1_inputs)
            position2_feature = self.position2_emb(position2_inputs)
            word_list.append(position1_feature)
            word_list.append(position2_feature)


        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent

class HiddenLayer(nn.Module):
    def __init__(self, data, input_size, output_size, att_size):
        super(HiddenLayer, self).__init__()

        self.gpu = data.HP_gpu

        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = True
        self.lstm_layer = 1

        self.input_size = input_size

        if self.bilstm_flag:
            lstm_hidden = output_size // 2
        else:
            lstm_hidden = output_size


        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        self.attn = DotAttentionLayer(output_size, self.gpu)

        self.decrease_dim = nn.Linear(output_size, att_size, bias=False)

        if torch.cuda.is_available():
            self.droplstm = self.droplstm.cuda(self.gpu)
            self.lstm = self.lstm.cuda(self.gpu)
            self.attn = self.attn.cuda(self.gpu)
            self.decrease_dim = self.decrease_dim.cuda(self.gpu)



    def forward(self, word_represent, word_seq_lengths):

        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = lstm_out.transpose(1, 0)
        ## lstm_out (seq_len, seq_len, hidden_size)

        att_out = self.attn((lstm_out, word_seq_lengths))
        att_out = self.decrease_dim(att_out)

        lstm_out = self.droplstm(lstm_out)
        ## lstm_out (batch_size, seq_len, hidden_size)


        return lstm_out, att_out


class StitchUnit(nn.Module):
    def __init__(self, data, input_size):
        super(StitchUnit, self).__init__()

        self.gpu = data.HP_gpu

        self.input_size = input_size

        self.stitch = nn.Linear(input_size, input_size, bias=False)

        if torch.cuda.is_available():
            self.stitch = self.stitch.cuda(self.gpu)

    def forward(self, A, B):

        a_b = self.stitch(torch.cat((A,B), 1))

        a, b = torch.chunk(a_b, 2, 1)

        return a, b



class SeqModel(nn.Module):
    def __init__(self, data, input_size):
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        # data.label_alphabet_size += 2
        # self.word_hidden = WordSequence(data, False, True, data.use_char)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(input_size, label_size+2)

        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

        if torch.cuda.is_available():
            self.hidden2tag = self.hidden2tag.cuda(self.gpu)

        self.frozen = False


    def neg_log_likelihood_loss(self, hidden, batch_label, mask):

        outs = self.hidden2tag(hidden)

        batch_size = hidden.size(0)
        seq_len = hidden.size(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)

        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq


    def forward(self, hidden, mask):

        outs = self.hidden2tag(hidden)

        batch_size = hidden.size(0)
        seq_len = hidden.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq  = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return tag_seq



    def decode_nbest(self, hidden, mask, nbest):
        if not self.use_crf:
            print "Nbest output is currently supported only for CRF! Exit..."
            exit(0)

        outs = self.hidden2tag(hidden)

        batch_size = hidden.size(0)
        seq_len = hidden.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq


class ClassifyModel(nn.Module):
    def __init__(self, data, input_size):
        super(ClassifyModel, self).__init__()

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss

        relation_alphabet_id = data.re_feature_name2id['[RELATION]']
        label_size = data.re_feature_alphabet_sizes[relation_alphabet_id]

        self.attn = DotAttentionLayer(input_size, self.gpu)

        # instance-level feature
        entity_type_alphabet_id = data.re_feature_name2id['[ENTITY_TYPE]']
        self.entity_type_emb = nn.Embedding(data.re_feature_alphabets[entity_type_alphabet_id].size(),
                                       data.re_feature_emb_dims[entity_type_alphabet_id], data.pad_idx)
        self.entity_type_emb.weight.data.copy_(
                torch.from_numpy(my_utils.random_embedding(data.re_feature_alphabets[entity_type_alphabet_id].size(),
                                                           data.re_feature_emb_dims[entity_type_alphabet_id])))

        entity_alphabet_id = data.re_feature_name2id['[ENTITY]']
        self.entity_emb = nn.Embedding(data.re_feature_alphabets[entity_alphabet_id].size(),
                                       data.re_feature_emb_dims[entity_alphabet_id], data.pad_idx)
        self.entity_emb.weight.data.copy_(
                torch.from_numpy(my_utils.random_embedding(data.re_feature_alphabets[entity_alphabet_id].size(),
                                                           data.re_feature_emb_dims[entity_alphabet_id])))

        self.dot_att = DotAttentionLayer(data.re_feature_emb_dims[entity_alphabet_id], data.HP_gpu)

        tok_num_alphabet_id = data.re_feature_name2id['[TOKEN_NUM]']
        self.tok_num_betw_emb = nn.Embedding(data.re_feature_alphabets[tok_num_alphabet_id].size(),
                                       data.re_feature_emb_dims[tok_num_alphabet_id], data.pad_idx)
        self.tok_num_betw_emb.weight.data.copy_(
                torch.from_numpy(my_utils.random_embedding(data.re_feature_alphabets[tok_num_alphabet_id].size(),
                                                           data.re_feature_emb_dims[tok_num_alphabet_id])))

        et_num_alphabet_id = data.re_feature_name2id['[ENTITY_NUM]']
        self.et_num_emb = nn.Embedding(data.re_feature_alphabets[et_num_alphabet_id].size(),
                                       data.re_feature_emb_dims[et_num_alphabet_id], data.pad_idx)
        self.et_num_emb.weight.data.copy_(
                torch.from_numpy(my_utils.random_embedding(data.re_feature_alphabets[et_num_alphabet_id].size(),
                                                           data.re_feature_emb_dims[et_num_alphabet_id])))

        self.input_size = input_size + 2 * data.re_feature_emb_dims[entity_type_alphabet_id] + 2 * data.re_feature_emb_dims[entity_alphabet_id] + \
                          data.re_feature_emb_dims[tok_num_alphabet_id] + data.re_feature_emb_dims[et_num_alphabet_id]

        self.linear = nn.Linear(self.input_size, label_size, bias=False)

        self.loss_function = nn.NLLLoss(size_average=self.average_batch)

        self.frozen = False

        if torch.cuda.is_available():
            self.attn = self.attn.cuda(data.HP_gpu)
            self.entity_type_emb = self.entity_type_emb.cuda(data.HP_gpu)
            self.entity_emb = self.entity_emb.cuda(data.HP_gpu)
            self.dot_att = self.dot_att.cuda(data.HP_gpu)
            self.tok_num_betw_emb = self.tok_num_betw_emb.cuda(data.HP_gpu)
            self.et_num_emb = self.et_num_emb.cuda(data.HP_gpu)
            self.linear = self.linear.cuda(data.HP_gpu)


    def neg_log_likelihood_loss(self, hidden, word_seq_lengths, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num, targets):

        hidden_features = self.attn((hidden, word_seq_lengths))

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

        v_et_num = self.et_num_emb(et_num)

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)

        outs = self.linear(x)

        score = F.log_softmax(outs, 1)
        total_loss = self.loss_function(score, targets)
        _, tag_seq = torch.max(score, 1)
        return total_loss, tag_seq


    def forward(self, hidden, word_seq_lengths, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num):

        hidden_features = self.attn((hidden, word_seq_lengths))

        e1_t = self.entity_type_emb(e1_type)
        e2_t = self.entity_type_emb(e2_type)

        e1 = self.entity_emb(e1_token)
        e1 = self.dot_att((e1, e1_length))
        e2 = self.entity_emb(e2_token)
        e2 = self.dot_att((e2, e2_length))

        v_tok_num_betw = self.tok_num_betw_emb(tok_num_betw)

        v_et_num = self.et_num_emb(et_num)

        x = torch.cat((hidden_features, e1_t, e2_t, e1, e2, v_tok_num_betw, v_et_num), dim=1)

        outs = self.linear(x)


        _, tag_seq = torch.max(outs, 1)

        return tag_seq



def train(data, ner_dir, re_dir):


    ner_wordrep = WordRep(data, False, True, True, data.use_char)
    ner_hiddenlist = []
    output_size = data.HP_hidden_dim
    att_size = opt.att_size
    stitch_list = []

    for i in range(opt.hidden_num):
        if i == 0:
            input_size = data.word_emb_dim+data.HP_char_hidden_dim+data.feature_emb_dims[data.feature_name2id['[Cap]']]+ \
                         data.feature_emb_dims[data.feature_name2id['[POS]']]

        else:
            input_size = output_size + att_size

        temp = HiddenLayer(data, input_size, output_size, att_size)
        ner_hiddenlist.append(temp)

        temp = StitchUnit(data, att_size*2)
        stitch_list.append(temp)

    seq_model = SeqModel(data, output_size + att_size)

    re_wordrep = WordRep(data, True, False, True, False)
    re_hiddenlist = []
    for i in range(opt.hidden_num):
        if i==0:
            input_size = data.word_emb_dim + data.feature_emb_dims[data.feature_name2id['[POS]']]+\
                         2*data.re_feature_emb_dims[data.re_feature_name2id['[POSITION]']]
        else:
            input_size = output_size + att_size

        temp = HiddenLayer(data, input_size, output_size, att_size)
        re_hiddenlist.append(temp)

    classify_model = ClassifyModel(data, output_size + att_size)

    iter_parameter = itertools.chain(*map(list, [ner_wordrep.parameters(), seq_model.parameters()]+[f.parameters() for f in ner_hiddenlist]))
    ner_optimizer = optim.Adam(iter_parameter, lr=data.HP_lr, weight_decay=data.HP_l2)
    iter_parameter = itertools.chain(*map(list, [re_wordrep.parameters(), classify_model.parameters()]+[f.parameters() for f in re_hiddenlist]))
    re_optimizer = optim.Adam(iter_parameter, lr=data.HP_lr, weight_decay=data.HP_l2)


    if data.tune_wordemb == False:
        my_utils.freeze_net(ner_wordrep.word_embedding)
        my_utils.freeze_net(re_wordrep.word_embedding)


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

        ner_wordrep.train()
        ner_wordrep.zero_grad()
        for hidden_layer in ner_hiddenlist:
            hidden_layer.train()
            hidden_layer.zero_grad()
        seq_model.train()
        seq_model.zero_grad()

        re_wordrep.train()
        re_wordrep.zero_grad()
        for hidden_layer in re_hiddenlist:
            hidden_layer.train()
            hidden_layer.zero_grad()
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

            if batch_id < min_batch:
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > ner_train_num:
                    end = ner_train_num
                instance = data.train_Ids[start:end]
                ner_batch_word, ner_batch_features, ner_batch_wordlen, ner_batch_wordrecover, ner_batch_char, ner_batch_charlen, \
                ner_batch_charrecover, ner_batch_label, ner_mask, ner_batch_permute_label = batchify_with_label(instance, data.HP_gpu)

                [re_batch_word, re_batch_features, re_batch_wordlen, re_batch_wordrecover, re_batch_char, re_batch_charlen,
                 re_batch_charrecover, re_position1_seq_tensor, re_position2_seq_tensor, re_e1_token, re_e1_length, re_e2_token, re_e2_length,
                 re_e1_type, re_e2_type, re_tok_num_betw, re_et_num], [re_targets, re_targets_permute] = \
                    my_utils.endless_get_next_batch_without_rebatch1(re_train_loader, re_train_iter)

                if ner_batch_word.size(0) != re_batch_word.size(0):
                    continue # if batch size is not equal, we ignore such batch

                ner_word_rep = ner_wordrep.forward(ner_batch_word, ner_batch_features, ner_batch_wordlen, ner_batch_char, ner_batch_charlen,
                                                   ner_batch_charrecover, None, None)

                re_word_rep = re_wordrep.forward(re_batch_word, re_batch_features, re_batch_wordlen, re_batch_char, re_batch_charlen,
                                                 re_batch_charrecover, re_position1_seq_tensor, re_position2_seq_tensor)

                ner_hidden = ner_word_rep
                re_hidden = re_word_rep
                for i in range(opt.hidden_num):

                    ner_lstm_out, ner_att_out = ner_hiddenlist[i].forward(ner_hidden, ner_batch_wordlen)
                    re_lstm_out, re_att_out = re_hiddenlist[i].forward(re_hidden, re_batch_wordlen)

                    ner_att_out, re_att_out = stitch_list[i].forward(ner_att_out, re_att_out)

                    ner_hidden = torch.cat((ner_lstm_out, ner_att_out.unsqueeze(1).expand(-1, ner_lstm_out.size(1), -1)), 2)
                    re_hidden = torch.cat((re_lstm_out, re_att_out.unsqueeze(1).expand(-1, re_lstm_out.size(1), -1)), 2)


                ner_loss, ner_tag_seq = seq_model.neg_log_likelihood_loss(ner_hidden, ner_batch_label, ner_mask)
                re_loss, re_pred = classify_model.neg_log_likelihood_loss(re_hidden, re_batch_wordlen,
                                                                    re_e1_token, re_e1_length, re_e2_token, re_e2_length, re_e1_type,
                                                                    re_e2_type, re_tok_num_betw, re_et_num, re_targets)

                ner_loss.backward(retain_graph=True)
                re_loss.backward()

                ner_optimizer.step()
                re_optimizer.step()

                ner_wordrep.zero_grad()
                for hidden_layer in ner_hiddenlist:
                    hidden_layer.zero_grad()
                seq_model.zero_grad()

                re_wordrep.zero_grad()
                for hidden_layer in re_hiddenlist:
                    hidden_layer.zero_grad()
                classify_model.zero_grad()

            else:

                if batch_id < ner_total_batch:
                    start = batch_id * batch_size
                    end = (batch_id + 1) * batch_size
                    if end > ner_train_num:
                        end = ner_train_num
                    instance = data.train_Ids[start:end]
                    batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, \
                        batch_permute_label = batchify_with_label(instance, data.HP_gpu)

                    ner_word_rep = ner_wordrep.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                                 batch_charrecover, None, None)

                    ner_hidden = ner_word_rep
                    for i in range(opt.hidden_num):
                        ner_lstm_out, ner_att_out = ner_hiddenlist[i].forward(ner_hidden, batch_wordlen)

                        ner_hidden = torch.cat((ner_lstm_out, ner_att_out.unsqueeze(1).expand(-1, ner_lstm_out.size(1), -1)), 2)


                    loss, tag_seq = seq_model.neg_log_likelihood_loss(ner_hidden, batch_label, mask)

                    loss.backward()
                    ner_optimizer.step()
                    ner_wordrep.zero_grad()
                    for hidden_layer in ner_hiddenlist:
                        hidden_layer.zero_grad()
                    seq_model.zero_grad()



                if batch_id < re_total_batch:
                    [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
                     batch_char, batch_charlen, batch_charrecover, \
                     position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
                     tok_num_betw, et_num], [targets, targets_permute] = my_utils.endless_get_next_batch_without_rebatch1(
                        re_train_loader, re_train_iter)


                    re_word_rep = re_wordrep.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                                      batch_charrecover, position1_seq_tensor, position2_seq_tensor)

                    re_hidden = re_word_rep
                    for i in range(opt.hidden_num):
                        re_lstm_out, re_att_out = re_hiddenlist[i].forward(re_hidden, batch_wordlen)

                        re_hidden = torch.cat((re_lstm_out, re_att_out.unsqueeze(1).expand(-1, re_lstm_out.size(1), -1)), 2)

                    loss, pred = classify_model.neg_log_likelihood_loss(re_hidden, batch_wordlen,
                                                                        e1_token, e1_length, e2_token, e2_length, e1_type,
                                                                        e2_type, tok_num_betw, et_num, targets)
                    loss.backward()
                    re_optimizer.step()
                    re_wordrep.zero_grad()
                    for hidden_layer in re_hiddenlist:
                        hidden_layer.zero_grad()
                    classify_model.zero_grad()




        epoch_finish = time.time()
        print("epoch: %s training finished. Time: %.2fs" % (idx, epoch_finish - epoch_start))

        ner_score = ner_evaluate(data, ner_wordrep, ner_hiddenlist, seq_model, "test")
        print("ner evaluate: f: %.4f" % (ner_score))

        re_score = re_evaluate(re_wordrep, re_hiddenlist, classify_model, re_test_loader)
        print("re evaluate: f: %.4f" % (re_score))

        if ner_score+re_score > best_ner_score+best_re_score:
            print("new best score: ner: %.4f , re: %.4f" % (ner_score, re_score))
            best_ner_score = ner_score
            best_re_score = re_score

            torch.save(ner_wordrep.state_dict(), os.path.join(ner_dir, 'wordrep.pkl'))
            for i, hidden_layer in enumerate(ner_hiddenlist):
                torch.save(hidden_layer.state_dict(), os.path.join(ner_dir, 'hidden_{}.pkl'.format(i)))
                torch.save(stitch_list[i].state_dict(), os.path.join(ner_dir, 'stitch_{}.pkl'.format(i)))
            torch.save(seq_model.state_dict(), os.path.join(ner_dir, 'model.pkl'))

            torch.save(re_wordrep.state_dict(), os.path.join(re_dir, 'wordrep.pkl'))
            for i, hidden_layer in enumerate(re_hiddenlist):
                torch.save(hidden_layer.state_dict(), os.path.join(re_dir, 'hidden_{}.pkl'.format(i)))
            torch.save(classify_model.state_dict(), os.path.join(re_dir, 'model.pkl'))

def ner_evaluate(data, wordrep, hiddenlist, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print "Error: wrong evaluate name,", name

    wordrep.eval()
    for hidden in hiddenlist:
        hidden.eval()
    model.eval()
    batch_size = data.HP_batch_size
    att_size = opt.att_size

    correct = 0
    total = 0

    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue

        with torch.no_grad():
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, _ \
                = batchify_with_label(instance, data.HP_gpu, True)

            ner_word_rep = wordrep.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,batch_charrecover, None, None)

            ner_hidden = ner_word_rep
            for i in range(opt.hidden_num):
                ner_lstm_out, ner_att_out = hiddenlist[i].forward(ner_hidden, batch_wordlen)

                ner_hidden = torch.cat((ner_lstm_out, ner_att_out.unsqueeze(1).expand(-1, ner_lstm_out.size(1), -1)), 2)

            tag_seq = model(ner_hidden, mask)


        for idx in range(mask.shape[0]):
            for idy in range(mask.shape[1]):
                if mask[idx][idy] != 0:
                    total += 1
                    if tag_seq[idx][idy] == batch_label[idx][idy]:
                        correct += 1


    acc = 1.0 * correct / total
    return acc

def re_evaluate(wordrep, hiddenlist, model, loader):
    wordrep.eval()
    for hidden in hiddenlist:
        hidden.eval()
    model.eval()
    it = iter(loader)
    correct = 0
    total = 0
    att_size = opt.att_size

    for [batch_word, batch_features, batch_wordlen, batch_wordrecover, \
            batch_char, batch_charlen, batch_charrecover, \
            position1_seq_tensor, position2_seq_tensor, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, \
            tok_num_betw, et_num], [targets, targets_permute] in it:


        with torch.no_grad():
            re_word_rep = wordrep.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                             batch_charrecover, position1_seq_tensor, position2_seq_tensor)

            re_hidden = re_word_rep
            for i in range(opt.hidden_num):
                re_lstm_out, re_att_out = hiddenlist[i].forward(re_hidden, batch_wordlen)

                re_hidden = torch.cat((re_lstm_out, re_att_out.unsqueeze(1).expand(-1, re_lstm_out.size(1), -1)), 2)

            pred = model.forward(re_hidden, batch_wordlen, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num)

            total += targets.size(0)
            correct += (pred == targets).sum().data.item()

    # acc = 100.0 * correct / total
    acc = 1.0 * correct / total
    return acc

def test(data, opt, predict_dir):
    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(data.test_dir)

    # evaluate on test data and output results in bioc format, one doc one file

    data.load(opt.data_file)
    data.MAX_SENTENCE_LENGTH = -1
    data.show_data_summary()

    output_size = data.HP_hidden_dim
    att_size = opt.att_size

    data.fix_alphabet()
    seq_model = SeqModel(data, output_size+att_size)
    seq_model.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'model.pkl')))

    ner_hiddenlist = []

    for i in range(opt.hidden_num):
        if i == 0:
            input_size = data.word_emb_dim+data.HP_char_hidden_dim+data.feature_emb_dims[data.feature_name2id['[Cap]']]+ \
                         data.feature_emb_dims[data.feature_name2id['[POS]']]

        else:
            input_size = output_size + att_size

        temp = HiddenLayer(data, input_size, output_size, att_size)
        temp.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'hidden_{}.pkl'.format(i))))
        ner_hiddenlist.append(temp)

    ner_wordrep = WordRep(data, False, True, True, data.use_char)
    ner_wordrep.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'wordrep.pkl')))

    classify_model = ClassifyModel(data, output_size+att_size)
    classify_model.load_state_dict(torch.load(os.path.join(opt.re_dir, 'model.pkl')))
    re_hiddenlist = []

    for i in range(opt.hidden_num):
        if i==0:
            input_size = data.word_emb_dim + data.feature_emb_dims[data.feature_name2id['[POS]']]+\
                         2*data.re_feature_emb_dims[data.re_feature_name2id['[POSITION]']]
        else:
            input_size = output_size + att_size

        temp = HiddenLayer(data, input_size, output_size, att_size)
        temp.load_state_dict(torch.load(os.path.join(opt.re_dir, 'hidden_{}.pkl'.format(i))))
        re_hiddenlist.append(temp)

    re_wordrep = WordRep(data, True, False, True, False)
    re_wordrep.load_state_dict(torch.load(os.path.join(opt.re_dir, 'wordrep.pkl')))

    for i in tqdm(range(len(test_name))):
        doc_name = test_name[i]
        doc_token = test_token[i]
        doc_entity = test_entity[i]

        if opt.use_gold_ner:
            entities = []
            for _, e in doc_entity.iterrows():
                entity = Entity()
                entity.create(e['id'], e['type'], e['start'], e['end'], e['text'], e['sent_idx'], e['tf_start'], e['tf_end'])
                entities.append(entity)
        else:

            ncrf_data = ner.generateDataForOneDoc(doc_token, doc_entity)

            data.raw_texts, data.raw_Ids = ner.read_instanceFromBuffer(ncrf_data, data.word_alphabet, data.char_alphabet,
                                                         data.feature_alphabets, data.label_alphabet, data.number_normalized,
                                                         data.MAX_SENTENCE_LENGTH)


            decode_results = ner_evaluateWhenTest(data, ner_wordrep, ner_hiddenlist, seq_model)


            entities = ner.translateNCRFPPintoEntities(doc_token, decode_results, doc_name)



        collection = bioc.BioCCollection()
        document = bioc.BioCDocument()
        collection.add_document(document)
        document.id = doc_name
        passage = bioc.BioCPassage()
        document.add_passage(passage)
        passage.offset = 0

        for entity in entities:
            anno_entity = bioc.BioCAnnotation()
            passage.add_annotation(anno_entity)
            anno_entity.id = entity.id
            anno_entity.infons['type'] = entity.type
            anno_entity_location = bioc.BioCLocation(entity.start, entity.getlength())
            anno_entity.add_location(anno_entity_location)
            anno_entity.text = entity.text


        test_X, test_other = relation_extraction.getRelationInstanceForOneDoc(doc_token, entities, doc_name, data)

        relations = re_evaluateWhenTest(re_wordrep, re_hiddenlist, classify_model, test_X, data, test_other, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']])

        for relation in relations:
            bioc_relation = bioc.BioCRelation()
            passage.add_relation(bioc_relation)
            bioc_relation.id = relation.id
            bioc_relation.infons['type'] = relation.type

            node1 = bioc.BioCNode(relation.node1.id, 'annotation 1')
            bioc_relation.add_node(node1)
            node2 = bioc.BioCNode(relation.node2.id, 'annotation 2')
            bioc_relation.add_node(node2)


        with open(os.path.join(predict_dir, doc_name + ".bioc.xml"), 'w') as fp:
            bioc.dump(collection, fp)



def ner_evaluateWhenTest(data, wordrep, hiddenlist, model):

    instances = data.raw_Ids
    nbest_pred_results = []
    wordrep.eval()
    for hidden in hiddenlist:
        hidden.eval()
    model.eval()
    batch_size = data.HP_batch_size
    att_size = opt.att_size

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
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, _  \
            = batchify_with_label(instance, data.HP_gpu, True)

        ner_word_rep = wordrep.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                       batch_charrecover, None, None)

        ner_hidden = ner_word_rep
        for i in range(opt.hidden_num):
            ner_lstm_out, ner_att_out = hiddenlist[i].forward(ner_hidden, batch_wordlen)

            ner_hidden = torch.cat((ner_lstm_out, ner_att_out.unsqueeze(1).expand(-1, ner_lstm_out.size(1), -1)), 2)

        scores, nbest_tag_seq = model.decode_nbest(ner_hidden, mask, data.nbest)

        nbest_pred_result = ner.recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
        nbest_pred_results += nbest_pred_result

    return nbest_pred_results


def re_evaluateWhenTest(wordrep, hiddenlist, model, instances, data, test_other, relationVocab):
    wordrep.eval()
    for hidden in hiddenlist:
        hidden.eval()
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
            re_word_rep = wordrep.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                             batch_charrecover, position1_seq_tensor, position2_seq_tensor)

            re_hidden = re_word_rep
            for i in range(opt.hidden_num):
                re_lstm_out, re_att_out = hiddenlist[i].forward(re_hidden, batch_wordlen)

                re_hidden = torch.cat((re_lstm_out, re_att_out.unsqueeze(1).expand(-1, re_lstm_out.size(1), -1)), 2)

            pred = model.forward(re_hidden, batch_wordlen, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num)

            pred = pred.index_select(0, batch_wordrecover)


        for i in range(start,end):

            former = test_other[i][0]
            latter = test_other[i][1]

            relation_type = relationVocab.get_instance(pred[i-start].item())
            if relation_type == '</unk>':
                continue
            elif relation_extraction.relationConstraint1(relation_type, former.type, latter.type) == False:
                continue
            else:
                relation = Relation()
                relation.create(str(relation_id), relation_type, former, latter)
                relations.append(relation)

                relation_id += 1

    return relations


def test_use_cotype(data, opt, test_file):
    test_token, test_entity, test_relation, test_name = preprocess_cotype.loadPreprocessData(test_file)

    # evaluate on test data and output results in bioc format, one doc one file

    data.load(opt.data_file)
    data.MAX_SENTENCE_LENGTH = -1
    data.show_data_summary()

    output_size = data.HP_hidden_dim
    att_size = opt.att_size

    data.fix_alphabet()
    seq_model = SeqModel(data, output_size+att_size)
    seq_model.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'model.pkl')))

    ner_hiddenlist = []

    for i in range(opt.hidden_num):
        if i == 0:
            input_size = data.word_emb_dim+data.HP_char_hidden_dim+data.feature_emb_dims[data.feature_name2id['[Cap]']]+ \
                         data.feature_emb_dims[data.feature_name2id['[POS]']]

        else:
            input_size = output_size + att_size

        temp = HiddenLayer(data, input_size, output_size, att_size)
        temp.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'hidden_{}.pkl'.format(i))))
        ner_hiddenlist.append(temp)

    ner_wordrep = WordRep(data, False, True, True, data.use_char)
    ner_wordrep.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'wordrep.pkl')))

    classify_model = ClassifyModel(data, output_size+att_size)
    classify_model.load_state_dict(torch.load(os.path.join(opt.re_dir, 'model.pkl')))
    re_hiddenlist = []

    for i in range(opt.hidden_num):
        if i==0:
            input_size = data.word_emb_dim + data.feature_emb_dims[data.feature_name2id['[POS]']]+\
                         2*data.re_feature_emb_dims[data.re_feature_name2id['[POSITION]']]
        else:
            input_size = output_size + att_size

        temp = HiddenLayer(data, input_size, output_size, att_size)
        temp.load_state_dict(torch.load(os.path.join(opt.re_dir, 'hidden_{}.pkl'.format(i))))
        re_hiddenlist.append(temp)

    re_wordrep = WordRep(data, True, False, True, False)
    re_wordrep.load_state_dict(torch.load(os.path.join(opt.re_dir, 'wordrep.pkl')))


    total_ner = 0
    correct_ner = 0
    predict_ner = 0

    total_re = 0
    correct_re = 0
    predict_re = 0

    for i in tqdm(range(len(test_name))):
        doc_name = test_name[i]
        doc_token = test_token[i]
        doc_entity = test_entity[i]
        doc_relation = test_relation[i]

        if opt.use_gold_ner:
            entities = []
            for _, e in doc_entity.iterrows():
                entity = Entity()
                entity.create(e['id'], e['type'], e['start'], e['end'], e['text'], e['sent_idx'], e['tf_start'], e['tf_end'])
                entities.append(entity)
        else:

            ncrf_data = test_cotype.generateDataForOneDoc(doc_token, doc_entity)

            data.raw_texts, data.raw_Ids = ner.read_instanceFromBuffer(ncrf_data, data.word_alphabet, data.char_alphabet,
                                                         data.feature_alphabets, data.label_alphabet, data.number_normalized,
                                                         data.MAX_SENTENCE_LENGTH)


            decode_results = ner_evaluateWhenTest(data, ner_wordrep, ner_hiddenlist, seq_model)


            entities = test_cotype.translateNCRFPPintoEntities(doc_token, decode_results, doc_name)


        test_X, test_other = test_cotype.getRelationInstanceForOneDoc(doc_token, entities, doc_name, data)

        relations = re_evaluateWhenTest_cotype(re_wordrep, re_hiddenlist, classify_model, test_X, data, test_other, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']])

        # evaluation
        predict_ner += len(entities)
        total_ner += doc_entity.shape[0]
        for predict in entities:
            for _, gold in doc_entity.iterrows():
                if gold['type'] == predict.type and gold['start'] == predict.start and gold['end'] == predict.end:
                    correct_ner += 1
                    break

        predict_re += len(relations)
        gold_relations = []
        for _, gold in doc_relation.iterrows():
            if gold['type'] == 'None':  # we don't count None relations
                continue
            gold_relation = Relation()
            gold_relation.type = gold['type']
            node1 = Entity()
            node1.text = gold['entity1_text']
            gold_relation.node1 = node1
            node2 = Entity()
            node2.text = gold['entity2_text']
            gold_relation.node2 = node2

            gold_relations.append(gold_relation)

        total_re += len(gold_relations)

        for predict in relations:
            for gold in gold_relations:
                if predict.equals_cotype(gold):
                    correct_re += 1
                    break

    ner_p = correct_ner * 1.0 / predict_ner
    ner_r = correct_ner * 1.0 / total_ner
    ner_f1 = 2.0 * ner_p * ner_r / (ner_p + ner_r)
    print("NER p: %.4f | r: %.4f | f1: %.4f" % (ner_p, ner_r, ner_f1))

    re_p = correct_re * 1.0 / predict_re
    re_r = correct_re * 1.0 / total_re
    re_f1 = 2.0 * re_p * re_r / (re_p + re_r)
    print("RE p: %.4f | r: %.4f | f1: %.4f" % (re_p, re_r, re_f1))



def re_evaluateWhenTest_cotype(wordrep, hiddenlist, model, instances, data, test_other, relationVocab):
    wordrep.eval()
    for hidden in hiddenlist:
        hidden.eval()
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
            re_word_rep = wordrep.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                             batch_charrecover, position1_seq_tensor, position2_seq_tensor)

            re_hidden = re_word_rep
            for i in range(opt.hidden_num):
                re_lstm_out, re_att_out = hiddenlist[i].forward(re_hidden, batch_wordlen)

                re_hidden = torch.cat((re_lstm_out, re_att_out.unsqueeze(1).expand(-1, re_lstm_out.size(1), -1)), 2)

            pred = model.forward(re_hidden, batch_wordlen, e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num)

            pred = pred.index_select(0, batch_wordrecover)


        for i in range(start,end):

            former = test_other[i][0]
            latter = test_other[i][1]

            relation_type = relationVocab.get_instance(pred[i-start].item())
            if relation_type == '</unk>':
                continue
            else:
                relation = Relation()
                relation.create(str(relation_id), relation_type, former, latter)
                relations.append(relation)

                relation_id += 1

    return relations