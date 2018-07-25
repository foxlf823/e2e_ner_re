
import logging
import random
import numpy as np
import torch

from options import opt
from test import *
import ner
import relation_extraction
from utils.data import data
import joint_train
import shared_reg
import shared_soft
import shared_stitch


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


train_ner_file = os.path.join(data.train_dir, 'ner_instance.txt')
test_ner_file = os.path.join(data.test_dir, 'ner_instance.txt')
predict_dir = os.path.join(data.test_dir, "predicted")
if not os.path.exists(predict_dir):
    os.makedirs(predict_dir)

if not os.path.exists(opt.ner_dir):
    os.makedirs(opt.ner_dir)
if not os.path.exists(opt.re_dir):
    os.makedirs(opt.re_dir)

if opt.whattodo==1:
    # parsing original data into pandas
    # preprocess.preprocess(data.train_dir)
    # preprocess.preprocess(data.test_dir)

    # ner
    # generate crf++ style data
    train_token, train_entity, train_relation, train_name = preprocess.loadPreprocessData(data.train_dir)
    ner.generateData(train_token, train_entity, train_name, train_ner_file)
    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(data.test_dir)
    ner.generateData(test_token, test_entity, test_name, test_ner_file)
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
    data.build_re_feature_alphabets(train_token, train_entity, train_relation)
    if data.full_data:
        data.build_re_feature_alphabets(test_token, test_entity, test_relation)
    data.fix_re_alphabet()
    # generate instance
    data.generate_re_instance('train', train_token, train_entity, train_relation, train_name)
    data.generate_re_instance('test', test_token, test_entity, test_relation, test_name)
    # build emb
    data.build_re_pretrain_emb()

    data.show_data_summary()
    data.save(opt.data_file)

elif opt.whattodo==2:
    # train ner model
    data.load(opt.data_file)
    data.HP_iteration = opt.ner_iter
    data.max_epoch = opt.re_iter
    data.HP_gpu = opt.gpu
    data.unk_ratio = opt.unk_ratio
    data.show_data_summary()

    if opt.shared == 'hard':
        joint_train.joint_train1(data, opt.ner_dir, opt.re_dir)
    elif opt.shared == 'reg':
        shared_reg.train(data, opt.ner_dir, opt.re_dir)
    elif opt.shared == 'soft':
        shared_soft.train(data, opt.ner_dir, opt.re_dir)
    elif opt.shared == 'stitch':
        shared_stitch.train(data, opt.ner_dir, opt.re_dir)
    else:

        if opt.mutual_adv == 'grad' or opt.mutual_adv == 'label':
            joint_train.joint_train(data, opt.ner_dir, opt.re_dir)
        elif opt.self_adv == 'grad' or opt.self_adv == 'label':
            if opt.ner_iter > 0:
                ner.train(data, opt.ner_dir)
            if opt.re_iter > 0:
                relation_extraction.train(data, opt.re_dir)
        else:
            # select one of the below method as the training method
            joint_train.joint_train(data, opt.ner_dir, opt.re_dir)

            # if opt.ner_iter > 0:
            #     ner.train(data, opt.ner_dir)
            #
            # if opt.re_iter > 0:
            #     relation_extraction.train1(data, opt.re_dir)


elif opt.whattodo==3:

    if opt.shared == 'hard':
        test1(data, opt, predict_dir)
    elif opt.shared == 'reg':
        shared_reg.test(data, opt, predict_dir)
    elif opt.shared == 'soft':
        shared_soft.test(data, opt, predict_dir)
    elif opt.shared == 'stitch':
        shared_stitch.test(data, opt, predict_dir)
    else:
        test(data, opt, predict_dir)


