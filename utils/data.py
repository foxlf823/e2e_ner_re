# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 13:58:10
import sys
import numpy as np
from alphabet import Alphabet
from functions import *
import cPickle as pickle
from options import opt
import my_utils1
import relation_extraction

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"

class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')

        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None
        self.feature_name2id = {}


        self.label_alphabet = Alphabet('label',True)
        self.tagScheme = "NoSeg" ## BMES/BIO
        
        self.seg = True

        ### I/O
        self.train_dir = None 
        self.dev_dir = None 
        self.test_dir = None


        self.model_dir = None ## model save  file


        self.word_emb_dir = None 
        self.char_emb_dir = None
        self.feature_emb_dirs = []

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []


        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []


        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30

        ###Networks
        self.word_feature_extractor = "LSTM" ## "LSTM"/"CNN"/"GRU"/
        self.use_char = True
        self.char_feature_extractor = "CNN" ## "LSTM"/"CNN"/"GRU"/None
        self.use_crf = True
        self.nbest = None
        
        ## Training
        self.average_batch_loss = False

        ### Hyperparameters
        self.HP_cnn_layer = 4
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        
        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

        # both
        self.full_data = False
        self.tune_wordemb = False

        # relation
        self.pretrain = None
        self.max_seq_len = 500
        self.pad_idx = 1
        self.sent_window = 3
        self.output =None
        self.unk_ratio=1
        self.seq_feature_size=256
        self.max_epoch = 100
        self.feature_extractor=None

        self.re_feature_name = []
        self.re_feature_name2id = {}
        self.re_feature_alphabets = []
        self.re_feature_num = len(self.re_feature_alphabets)
        self.re_feat_config = None

        self.re_train_X = []
        self.re_dev_X = []
        self.re_test_X = []
        self.re_train_Y = []
        self.re_dev_Y = []
        self.re_test_Y = []

        
    def show_data_summary(self):
        print("++"*50)
        print("DATA SUMMARY START:")
        print(" I/O:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding  dir: %s"%(self.word_emb_dir))
        print("     Char embedding  dir: %s"%(self.char_emb_dir))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Norm   word     emb: %s"%(self.norm_word_emb))
        print("     Norm   char     emb: %s"%(self.norm_char_emb))
        print("     Train  file directory: %s"%(self.train_dir))
        print("     Dev    file directory: %s"%(self.dev_dir))
        print("     Test   file directory: %s"%(self.test_dir))


        print("     Model  file directory: %s"%(self.model_dir))


        print("     Train instance number: %s"%(len(self.train_texts)))
        print("     Dev   instance number: %s"%(len(self.dev_texts)))
        print("     Test  instance number: %s"%(len(self.test_texts)))

        print("     FEATURE num: %s"%(self.feature_num))
        for idx in range(self.feature_num):
            print("         Fe: %s  alphabet  size: %s"%(self.feature_alphabets[idx].name, self.feature_alphabet_sizes[idx]))
            print("         Fe: %s  embedding  dir: %s"%(self.feature_alphabets[idx].name, self.feature_emb_dirs[idx]))
            print("         Fe: %s  embedding size: %s"%(self.feature_alphabets[idx].name, self.feature_emb_dims[idx]))
            print("         Fe: %s  norm       emb: %s"%(self.feature_alphabets[idx].name, self.norm_feature_embs[idx]))
        # for k, v in self.feat_config.items():
        #     print("         Feature: %s, size %s, norm %s, dir %s"%(k, v['emb_size'], v['emb_norm'], v['emb_dir']))

        print(" "+"++"*20)
        print(" Model Network:")
        print("     Model        use_crf: %s"%(self.use_crf))
        print("     Model word extractor: %s"%(self.word_feature_extractor))
        print("     Model       use_char: %s"%(self.use_char))
        if self.use_char:
            print("     Model char extractor: %s"%(self.char_feature_extractor))
            print("     Model char_hidden_dim: %s"%(self.HP_char_hidden_dim))
        print(" "+"++"*20)
        print(" Training:")
        print("     Optimizer: %s"%(self.optimizer))
        print("     Iteration: %s"%(self.HP_iteration))
        print("     BatchSize: %s"%(self.HP_batch_size))
        print("     Average  batch   loss: %s"%(self.average_batch_loss))

        print(" "+"++"*20)
        print(" Hyperparameters:")
        
        print("     Hyper              lr: %s"%(self.HP_lr))
        print("     Hyper        lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyper         HP_clip: %s"%(self.HP_clip))
        print("     Hyper        momentum: %s"%(self.HP_momentum))
        print("     Hyper              l2: %s"%(self.HP_l2))
        print("     Hyper      hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyper         dropout: %s"%(self.HP_dropout))
        print("     Hyper      lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyper          bilstm: %s"%(self.HP_bilstm))
        print("     Hyper             GPU: %s"%(self.HP_gpu))

        print(" " + "++" * 20)
        print(" Both:")

        print("     full data: %s" % (self.full_data))
        print("     Tune  word embeddings: %s" % (self.tune_wordemb))

        print(" "+"++"*20)
        print(" Relation:")

        print("     Pretrain directory: %s" % (self.pretrain))
        print("     max sequence length: %s" % (self.max_seq_len))
        print("     pad index: %s" % (self.pad_idx))
        print("     sentence window: %s" % (self.sent_window))
        print("     Output directory: %s" % (self.output))
        print("     The ratio using negative instnaces 0~1: %s" % (self.unk_ratio))
        print("     Size of seqeuence feature representation: %s" % (self.seq_feature_size))
        print("     Iteration for relation training: %s" % (self.max_epoch))
        print("     feature_extractor: %s" % (self.feature_extractor))

        print("     RE FEATURE num: %s"%(self.re_feature_num))
        for idx in range(self.re_feature_num):
            print("         Fe: %s  alphabet  size: %s"%(self.re_feature_alphabets[idx].name, self.re_feature_alphabet_sizes[idx]))
            print("         Fe: %s  embedding  dir: %s"%(self.re_feature_alphabets[idx].name, self.re_feature_emb_dirs[idx]))
            print("         Fe: %s  embedding size: %s"%(self.re_feature_alphabets[idx].name, self.re_feature_emb_dims[idx]))
            print("         Fe: %s  norm       emb: %s"%(self.re_feature_alphabets[idx].name, self.re_norm_feature_embs[idx]))

        print("     RE Train instance number: %s"%(len(self.re_train_Y)))
        print("     RE Dev   instance number: %s"%(len(self.re_dev_Y)))
        print("     RE Test  instance number: %s"%(len(self.re_test_Y)))

        print("DATA SUMMARY END.")
        print("++"*50)
        sys.stdout.flush()


    def initial_feature_alphabets(self, input_file):
        items = open(input_file,'r').readline().strip('\n').split()
        total_column = len(items)
        if total_column > 2:
            id = 0
            for idx in range(1, total_column-1):
                feature_prefix = items[idx].split(']',1)[0]+"]"
                self.feature_alphabets.append(Alphabet(feature_prefix))
                self.feature_name.append(feature_prefix)
                self.feature_name2id[feature_prefix] = id
                id += 1
                print "Find feature: ", feature_prefix 
        self.feature_num = len(self.feature_alphabets)
        self.pretrain_feature_embeddings = [None]*self.feature_num
        self.feature_emb_dims = [20]*self.feature_num
        self.feature_emb_dirs = [None]*self.feature_num 
        self.norm_feature_embs = [False]*self.feature_num
        self.feature_alphabet_sizes = [0]*self.feature_num
        if self.feat_config:
            for idx in range(self.feature_num):
                if self.feature_name[idx] in self.feat_config:
                    self.feature_emb_dims[idx] = self.feat_config[self.feature_name[idx]]['emb_size']
                    self.feature_emb_dirs[idx] = self.feat_config[self.feature_name[idx]]['emb_dir']
                    self.norm_feature_embs[idx] = self.feat_config[self.feature_name[idx]]['emb_norm']
        # exit(0)


    def build_alphabet(self, input_file):
        in_lines = open(input_file,'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0].decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                ## build feature alphabet 
                for idx in range(self.feature_num):
                    feat_idx = pairs[idx+1].split(']',1)[-1]
                    self.feature_alphabets[idx].add(feat_idx)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        for idx in range(self.feature_num):
            self.feature_alphabet_sizes[idx] = self.feature_alphabets[idx].size()
        startS = False
        startB = False
        for label,_ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close() 
        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()

    def initial_re_feature_alphabets(self):
        id = 0
        for k, v in self.re_feat_config.items():
            self.re_feature_alphabets.append(Alphabet(k))
            self.re_feature_name.append(k)
            self.re_feature_name2id[k] = id
            id += 1

        self.re_feature_num = len(self.re_feature_alphabets)
        self.re_pretrain_feature_embeddings = [None]*self.re_feature_num
        self.re_feature_emb_dims = [20]*self.re_feature_num
        self.re_feature_emb_dirs = [None]*self.re_feature_num
        self.re_norm_feature_embs = [False]*self.re_feature_num
        self.re_feature_alphabet_sizes = [0]*self.re_feature_num
        if self.re_feat_config:
            for idx in range(self.re_feature_num):
                if self.re_feature_name[idx] in self.re_feat_config:
                    self.re_feature_emb_dims[idx] = self.re_feat_config[self.re_feature_name[idx]]['emb_size']
                    self.re_feature_emb_dirs[idx] = self.re_feat_config[self.re_feature_name[idx]]['emb_dir']
                    self.re_norm_feature_embs[idx] = self.re_feat_config[self.re_feature_name[idx]]['emb_norm']


    def build_re_feature_alphabets(self, tokens, entities, relations):

        entity_type_alphabet = self.re_feature_alphabets[self.re_feature_name2id['[ENTITY_TYPE]']]
        entity_alphabet = self.re_feature_alphabets[self.re_feature_name2id['[ENTITY]']]
        relation_alphabet = self.re_feature_alphabets[self.re_feature_name2id['[RELATION]']]
        token_num_alphabet = self.re_feature_alphabets[self.re_feature_name2id['[TOKEN_NUM]']]
        entity_num_alphabet = self.re_feature_alphabets[self.re_feature_name2id['[ENTITY_NUM]']]
        position_alphabet = self.re_feature_alphabets[self.re_feature_name2id['[POSITION]']]

        for i, doc_token in enumerate(tokens):

            doc_entity = entities[i]
            doc_relation = relations[i]

            sent_idx = 0
            sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]
            while sentence.shape[0] != 0:

                entities_in_sentence = doc_entity[(doc_entity['sent_idx'] == sent_idx)]
                for _, entity in entities_in_sentence.iterrows():
                    entity_type_alphabet.add(entity['type'])
                    tk_idx = entity['tf_start']
                    while tk_idx <= entity['tf_end']:
                        entity_alphabet.add(
                            my_utils1.normalizeWord(sentence.iloc[tk_idx, 0]))  # assume 'text' is in 0 column
                        tk_idx += 1

                sent_idx += 1
                sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]

            for _, relation in doc_relation.iterrows():
                relation_alphabet.add(relation['type'])


        for i in range(data.max_seq_len):
            token_num_alphabet.add(i)
            entity_num_alphabet.add(i)
            position_alphabet.add(i)
            position_alphabet.add(-i)


        for idx in range(self.re_feature_num):
            self.re_feature_alphabet_sizes[idx] = self.re_feature_alphabets[idx].size()


    def fix_re_alphabet(self):
        for alphabet in self.re_feature_alphabets:
            alphabet.close()


    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm: %s, dir: %s"%(self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)
        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s"%(self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)
        for idx in range(self.feature_num):
            if self.feature_emb_dirs[idx]:
                print("Load pretrained feature %s embedding:, norm: %s, dir: %s"%(self.feature_name[idx], self.norm_feature_embs[idx], self.feature_emb_dirs[idx]))
                self.pretrain_feature_embeddings[idx], self.feature_emb_dims[idx] = build_pretrain_embedding(self.feature_emb_dirs[idx], self.feature_alphabets[idx], self.feature_emb_dims[idx], self.norm_feature_embs[idx])

    def build_re_pretrain_emb(self):
        for idx in range(self.re_feature_num):
            if self.re_feature_emb_dirs[idx]:
                print("Load pretrained re feature %s embedding:, norm: %s, dir: %s" % (self.re_feature_name[idx], self.re_norm_feature_embs[idx], self.re_feature_emb_dirs[idx]))
                self.re_pretrain_feature_embeddings[idx], self.re_feature_emb_dims[idx] = build_pretrain_embedding(
                    self.re_feature_emb_dirs[idx], self.re_feature_alphabets[idx], self.re_feature_emb_dims[idx],
                    self.re_norm_feature_embs[idx])

    def generate_instance(self, name, input_file):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(input_file, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))



    def generate_re_instance(self, name, tokens, entities, relations, names):
        self.fix_re_alphabet()
        if name == "train":
            self.re_train_X, self.re_train_Y = relation_extraction.getRelationInstance2(tokens, entities, relations, names, self)
        elif name == "dev":
            self.re_dev_X, self.re_dev_Y = relation_extraction.getRelationInstance2(tokens, entities, relations, names, self)
        elif name == "test":
            self.re_test_X, self.re_test_Y = relation_extraction.getRelationInstance2(tokens, entities, relations, names, self)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def load(self,data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self,save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()




    def read_config(self,config_file):
        config = config_file_to_dict(config_file)
        ## read data:
        the_item = 'train_dir'
        if the_item in config:
            self.train_dir = config[the_item]
        the_item = 'dev_dir'
        if the_item in config:
            self.dev_dir = config[the_item]
        the_item = 'test_dir'
        if the_item in config:
            self.test_dir = config[the_item]


        the_item = 'model_dir'
        if the_item in config:
            self.model_dir = config[the_item]


        the_item = 'word_emb_dir'
        if the_item in config:
            self.word_emb_dir = config[the_item]
        the_item = 'char_emb_dir'
        if the_item in config:
            self.char_emb_dir = config[the_item]


        the_item = 'MAX_SENTENCE_LENGTH'
        if the_item in config:
            self.MAX_SENTENCE_LENGTH = int(config[the_item])
        the_item = 'MAX_WORD_LENGTH'
        if the_item in config:
            self.MAX_WORD_LENGTH = int(config[the_item])

        the_item = 'norm_word_emb'
        if the_item in config:
            self.norm_word_emb = str2bool(config[the_item])
        the_item = 'norm_char_emb'
        if the_item in config:
            self.norm_char_emb = str2bool(config[the_item])
        the_item = 'number_normalized'
        if the_item in config:
            self.number_normalized = str2bool(config[the_item])


        the_item = 'seg'
        if the_item in config:
            self.seg = str2bool(config[the_item])
        the_item = 'word_emb_dim'
        if the_item in config:
            self.word_emb_dim = int(config[the_item])
        the_item = 'char_emb_dim'
        if the_item in config:
            self.char_emb_dim = int(config[the_item])

        ## read network:
        the_item = 'use_crf'
        if the_item in config:
            self.use_crf = str2bool(config[the_item])
        the_item = 'use_char'
        if the_item in config:
            self.use_char = str2bool(config[the_item])
        the_item = 'word_seq_feature'
        if the_item in config:
            self.word_feature_extractor = config[the_item]
        the_item = 'char_seq_feature'
        if the_item in config:
            self.char_feature_extractor = config[the_item]
        the_item = 'nbest'
        if the_item in config:
            self.nbest = int(config[the_item])

        the_item = 'feature'
        if the_item in config:
            self.feat_config = config[the_item] ## feat_config is a dict 






        ## read training setting:
        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]
        the_item = 'ave_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])


        ## read Hyperparameters:
        the_item = 'cnn_layer'
        if the_item in config:
            self.HP_cnn_layer = int(config[the_item])
        the_item = 'iteration'
        if the_item in config:
            self.HP_iteration = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.HP_batch_size = int(config[the_item])

        the_item = 'char_hidden_dim'
        if the_item in config:
            self.HP_char_hidden_dim = int(config[the_item])
        the_item = 'hidden_dim'
        if the_item in config:
            self.HP_hidden_dim = int(config[the_item])
        the_item = 'dropout'
        if the_item in config:
            self.HP_dropout = float(config[the_item])
        the_item = 'lstm_layer'
        if the_item in config:
            self.HP_lstm_layer = int(config[the_item])
        the_item = 'bilstm'
        if the_item in config:
            self.HP_bilstm = str2bool(config[the_item])

        the_item = 'gpu'
        if the_item in config:
            self.HP_gpu = int(config[the_item])
        the_item = 'learning_rate'
        if the_item in config:
            self.HP_lr = float(config[the_item])
        the_item = 'lr_decay'
        if the_item in config:
            self.HP_lr_decay = float(config[the_item])
        the_item = 'clip'
        if the_item in config:
            self.HP_clip = float(config[the_item])
        the_item = 'momentum'
        if the_item in config:
            self.HP_momentum = float(config[the_item])
        the_item = 'l2'
        if the_item in config:
            self.HP_l2 = float(config[the_item])

        # both
        the_item = 'full_data'
        if the_item in config:
            self.full_data = str2bool(config[the_item])

        the_item = 'tune_wordemb'
        if the_item in config:
            self.tune_wordemb = str2bool(config[the_item])

        # relation
        the_item = 'pretrain'
        if the_item in config:
            self.pretrain = config[the_item]

        the_item = 'max_seq_len'
        if the_item in config:
            self.max_seq_len = int(config[the_item])

        the_item = 'pad_idx'
        if the_item in config:
            self.pad_idx = int(config[the_item])

        the_item = 'sent_window'
        if the_item in config:
            self.sent_window = int(config[the_item])

        the_item = 'output'
        if the_item in config:
            self.output = config[the_item]

        the_item = 'unk_ratio'
        if the_item in config:
            self.unk_ratio = float(config[the_item])

        the_item = 'seq_feature_size'
        if the_item in config:
            self.seq_feature_size = int(config[the_item])

        the_item = 'max_epoch'
        if the_item in config:
            self.max_epoch = int(config[the_item])

        the_item = 'feature_extractor'
        if the_item in config:
            self.feature_extractor = config[the_item]

        the_item = 're_feature'
        if the_item in config:
            self.re_feat_config = config[the_item] ## feat_config is a dict


def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file,'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            pair = line.strip().split('#',1)[0].split('=',1)
            item = pair[0]
            if item=="feature" or item=='re_feature':
                if item not in config:
                    feat_dict = {}
                    config[item]= feat_dict 
                feat_dict = config[item]
                new_pair = pair[-1].split()
                feat_name = new_pair[0]
                one_dict = {}
                one_dict["emb_dir"] = None
                one_dict["emb_size"] = 10
                one_dict["emb_norm"] = False
                if len(new_pair) > 1:
                    for idx in range(1,len(new_pair)):
                        conf_pair = new_pair[idx].split('=')
                        if conf_pair[0] == "emb_dir":
                            one_dict["emb_dir"]=conf_pair[-1]
                        elif conf_pair[0] == "emb_size":
                            one_dict["emb_size"]=int(conf_pair[-1])
                        elif conf_pair[0] == "emb_norm":
                            one_dict["emb_norm"]=str2bool(conf_pair[-1])
                feat_dict[feat_name] = one_dict
                # print "feat",feat_dict
            else:
                if item in config:
                    print("Warning: duplicated config item found: %s, updated."%(pair[0]))
                config[item] = pair[-1]                
    return config


def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True 
    else:
        return False


data = Data()
data.read_config(opt.config)