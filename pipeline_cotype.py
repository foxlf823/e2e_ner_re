import logging
import random
import numpy as np
import torch
import os

from options import opt
import ner
import relation_extraction
from utils.data import data
import preprocess_cotype
import train_cotype
import test_cotype


logger = logging.getLogger()
if opt.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if opt.random_seed != 0:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)

logging.info(opt)

train_file = os.path.join(data.train_dir, 'train.json')
test_file = os.path.join(data.train_dir, 'test.json')
train_ner_file = os.path.join(data.train_dir, 'ner_train_instance.txt')
test_ner_file = os.path.join(data.train_dir, 'ner_test_instance.txt')

if not os.path.exists(opt.ner_dir):
    os.makedirs(opt.ner_dir)
if not os.path.exists(opt.re_dir):
    os.makedirs(opt.re_dir)



if opt.whattodo==1:
    # preprocess_cotype.statDataset(train_file, test_file)

    # preprocess_cotype.preprocess(train_file)
    # preprocess_cotype.preprocess(test_file)

    # ner
    # generate crf++ style data
    train_token, train_entity, train_relation, train_name = preprocess_cotype.loadPreprocessData(train_file)
    # preprocess_cotype.generateData(train_token, train_entity, train_name, train_ner_file)
    test_token, test_entity, test_relation, test_name = preprocess_cotype.loadPreprocessData(test_file)
    # preprocess_cotype.generateData(test_token, test_entity, test_name, test_ner_file)

    # build alphabet
    data.initial_feature_alphabets(train_ner_file)
    data.build_alphabet(train_ner_file)
    if data.full_data:
        data.build_alphabet(test_ner_file)
    data.fix_alphabet()

    # generate instance
    data.generate_instance('train', train_ner_file)
    data.generate_instance('test', test_ner_file)
    # build emb
    data.build_pretrain_emb()

    # re
    # generate alphabet
    data.initial_re_feature_alphabets()
    preprocess_cotype.build_re_feature_alphabets(data, train_token, train_entity, train_relation)
    if data.full_data:
        preprocess_cotype.build_re_feature_alphabets(data, test_token, test_entity, test_relation)
    data.fix_re_alphabet()
    # generate instance
    preprocess_cotype.generate_re_instance(data, 'train', train_token, train_entity, train_relation, train_name)
    preprocess_cotype.generate_re_instance(data, 'test', test_token, test_entity, test_relation, test_name)
    # build emb
    data.build_re_pretrain_emb()

    data.show_data_summary()
    data.save(opt.data_file)

elif opt.whattodo==2:

    data.load(opt.data_file)
    data.HP_iteration = opt.ner_iter
    data.max_epoch = opt.re_iter
    data.HP_gpu = opt.gpu
    data.unk_ratio = opt.unk_ratio
    data.show_data_summary()

    if opt.shared == 'hard':
        train_cotype.hard(data, opt.ner_dir, opt.re_dir)
    else:

        train_cotype.pipeline(data, opt.ner_dir, opt.re_dir)

elif opt.whattodo==3:

    if opt.shared == 'hard':
        test_cotype.hard(data, opt, test_file)
    else:
        test_cotype.pipeline(data, opt, test_file)