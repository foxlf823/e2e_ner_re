# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-03-30 16:20:07

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from wordsequence import WordSequence
from crf import CRF

class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf
        print "build network..."
        print "use_char: ", data.use_char 
        if data.use_char:
            print "char feature extractor: ", data.char_feature_extractor
        print "word feature extractor: ", data.word_feature_extractor
        print "use crf: ", self.use_crf

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        # data.label_alphabet_size += 2
        # self.word_hidden = WordSequence(data, False, True, data.use_char)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, label_size+2)

        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

        if torch.cuda.is_available():
            self.hidden2tag = self.hidden2tag.cuda(self.gpu)


    # def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        # outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, None, None)
    def neg_log_likelihood_loss(self, hidden, hidden_adv, batch_label, mask):
        if hidden_adv is not None:
            hidden = (hidden + hidden_adv)

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


    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)


    def decode_nbest(self, hidden, mask, nbest):
        if not self.use_crf:
            print "Nbest output is currently supported only for CRF! Exit..."
            exit(0)

        outs = self.hidden2tag(hidden)

        batch_size = hidden.size(0)
        seq_len = hidden.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

    def freeze_net(self):

        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_net(self):

        for p in self.parameters():
            p.requires_grad = True