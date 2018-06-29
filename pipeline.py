
import logging
import random
import numpy as np
import os
import shutil
import torch
from tqdm import tqdm

from options import opt
import preprocess
import ner
import relation_extraction
from utils.data import data
from model.seqmodel import SeqModel


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
data_file = data.model_dir + "/data"
model_file = data.model_dir + "/model"
output_dir = os.path.join(data.test_dir, "predicted")
re_output_dir = os.path.join(data.test_dir, "re_predicted")

if opt.whattodo==1:
    # parsing original data into pandas
    preprocess.preprocess(data.train_dir)
    preprocess.preprocess(data.test_dir)

    # prepare instances
    train_token, train_entity, train_relation, train_name = preprocess.loadPreprocessData(data.train_dir)
    ner.generateData(train_token, train_entity, train_name, train_ner_file)

    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(data.test_dir)
    ner.generateData(test_token, test_entity, test_name, test_ner_file)

    if not os.path.exists(data.pretrain):
        os.makedirs(data.pretrain)

    relation_extraction.pretrain(train_token, train_entity, train_relation, train_name, test_token, test_entity, test_relation,
                          test_name, data)

elif opt.whattodo==2:
    # step 2, train ner model
    # if not os.path.exists(data.model_dir):
    #     os.makedirs(data.model_dir)
    #
    # data.initial_feature_alphabets(train_ner_file)
    # data.build_alphabet(train_ner_file)
    # if data.full_data:
    #     data.build_alphabet(test_ner_file)
    # data.fix_alphabet()
    #
    # data.generate_instance('train', train_ner_file)
    # data.generate_instance('test', test_ner_file)
    #
    # data.build_pretrain_emb()
    #
    # data.show_data_summary()
    # save_data_name = data_file
    # data.save(save_data_name)
    #
    # ner.train(data, model_file)

    # train relation extraction model
    if not os.path.exists(data.output):
        os.makedirs(data.output)


    relation_extraction.train()

elif opt.whattodo==3:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # step 3, evaluate on test data and output results in bioc format, one doc one file

    # data.load(data_file)
    # data.MAX_SENTENCE_LENGTH = -1
    # data.show_data_summary()
    #
    # data.fix_alphabet()
    # model = SeqModel(data)
    # model.load_state_dict(torch.load(model_file))
    #
    # test_token, test_entity, _, test_name = preprocess.loadPreprocessData(data.test_dir)
    #
    # for i in tqdm(range(len(test_name))):
    #     doc_name = test_name[i]
    #     doc_token = test_token[i]
    #     doc_entity = test_entity[i]
    #
    #     ncrf_data = ner.generateDataForOneDoc(doc_token, doc_entity)
    #
    #     data.raw_texts, data.raw_Ids = ner.read_instanceFromBuffer(ncrf_data, data.word_alphabet, data.char_alphabet,
    #                                                  data.feature_alphabets, data.label_alphabet, data.number_normalized,
    #                                                  data.MAX_SENTENCE_LENGTH)
    #
    #     decode_results = ner.evaluateWhenTest(data, model)
    #
    #     ner.translateNCRFPPintoBioc(doc_token, decode_results, output_dir, doc_name)

    if not os.path.exists(re_output_dir):
        os.makedirs(re_output_dir)

    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(data.test_dir)

    relation_extraction.test2(test_token, test_entity, test_relation, test_name, re_output_dir)
