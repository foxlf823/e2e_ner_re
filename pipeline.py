
import logging
import random
import numpy as np
import os
import shutil
import torch
from tqdm import tqdm
import cPickle as pickle
import bioc

from options import opt
import preprocess
import ner
import relation_extraction
from utils.data import data
from model.seqmodel import SeqModel
import my_utils
import feature_extractor
from data_structure import *


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

if opt.whattodo==1:
    # parsing original data into pandas
    # preprocess.preprocess(data.train_dir)
    # preprocess.preprocess(data.test_dir)

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
    if not os.path.exists(data.model_dir):
        os.makedirs(data.model_dir)

    data.initial_feature_alphabets(train_ner_file)
    data.build_alphabet(train_ner_file)
    if data.full_data:
        data.build_alphabet(test_ner_file)
    data.fix_alphabet()

    data.generate_instance('train', train_ner_file)
    data.generate_instance('test', test_ner_file)

    data.build_pretrain_emb()

    data.show_data_summary()
    save_data_name = data_file
    data.save(save_data_name)

    ner.train(data, model_file)

    # train relation extraction model
    if not os.path.exists(data.output):
        os.makedirs(data.output)


    relation_extraction.train()

elif opt.whattodo==3:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(data.test_dir)

    # step 3, evaluate on test data and output results in bioc format, one doc one file

    data.load(data_file)
    data.MAX_SENTENCE_LENGTH = -1
    data.show_data_summary()

    data.fix_alphabet()
    model = SeqModel(data)
    model.load_state_dict(torch.load(model_file))

    logging.info("loading ... vocab")
    word_vocab = pickle.load(open(os.path.join(data.pretrain, 'word_vocab.pkl'), 'rb'))
    postag_vocab = pickle.load(open(os.path.join(data.pretrain, 'postag_vocab.pkl'), 'rb'))
    relation_vocab = pickle.load(open(os.path.join(data.pretrain, 'relation_vocab.pkl'), 'rb'))
    entity_type_vocab = pickle.load(open(os.path.join(data.pretrain, 'entity_type_vocab.pkl'), 'rb'))
    entity_vocab = pickle.load(open(os.path.join(data.pretrain, 'entity_vocab.pkl'), 'rb'))
    position_vocab1 = pickle.load(open(os.path.join(data.pretrain, 'position_vocab1.pkl'), 'rb'))
    position_vocab2 = pickle.load(open(os.path.join(data.pretrain, 'position_vocab2.pkl'), 'rb'))
    tok_num_betw_vocab = pickle.load(open(os.path.join(data.pretrain, 'tok_num_betw_vocab.pkl'), 'rb'))
    et_num_vocab = pickle.load(open(os.path.join(data.pretrain, 'et_num_vocab.pkl'), 'rb'))

    # cnnrnn
    if data.feature_extractor == 'lstm':
        m_low = feature_extractor.LSTMFeatureExtractor(word_vocab, postag_vocab, position_vocab1, position_vocab2,
                                                 1, data.seq_feature_size, data.HP_dropout)
    else:
        m_low = feature_extractor.CNNFeatureExtractor(word_vocab, postag_vocab, position_vocab1, position_vocab2,
                                            1, data.seq_feature_size,
                                            200, [3,4,5], data.HP_dropout)
    if torch.cuda.is_available():
        m_low = m_low.cuda(data.HP_gpu)
    m = feature_extractor.MLP(data.seq_feature_size, relation_vocab, entity_type_vocab, entity_vocab, tok_num_betw_vocab,
                                     et_num_vocab)
    if torch.cuda.is_available():
        m = m.cuda(data.HP_gpu)

    m_low.load_state_dict(torch.load(os.path.join(data.output, 'feature_extractor.pth')))
    m.load_state_dict(torch.load(os.path.join(data.output, 'model.pth')))


    for i in tqdm(range(len(test_name))):
        doc_name = test_name[i]
        doc_token = test_token[i]
        doc_entity = test_entity[i]

        ncrf_data = ner.generateDataForOneDoc(doc_token, doc_entity)

        data.raw_texts, data.raw_Ids = ner.read_instanceFromBuffer(ncrf_data, data.word_alphabet, data.char_alphabet,
                                                     data.feature_alphabets, data.label_alphabet, data.number_normalized,
                                                     data.MAX_SENTENCE_LENGTH)

        decode_results = ner.evaluateWhenTest(data, model)


        entities = ner.translateNCRFPPintoEntities(doc_token, decode_results, doc_name)
        # entities = []
        # for _, e in doc_entity.iterrows():
        #     entity = Entity()
        #     entity.create(e['id'], e['type'], e['start'], e['end'], e['text'], e['sent_idx'], e['tf_start'], e['tf_end'])
        #     entities.append(entity)


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



        test_X, test_other = my_utils.getRelationInstanceForOneDoc(doc_token, entities, doc_name,
                                                                   word_vocab, postag_vocab,
                                                                   relation_vocab, entity_type_vocab,
                                                                   entity_vocab, position_vocab1, position_vocab2,
                                                                   tok_num_betw_vocab, et_num_vocab)

        relations = relation_extraction.evaluateWhenTest(m_low, m, test_X, data, test_other, relation_vocab)

        for relation in relations:
            bioc_relation = bioc.BioCRelation()
            passage.add_relation(bioc_relation)
            bioc_relation.id = relation.id
            bioc_relation.infons['type'] = relation.type

            node1 = bioc.BioCNode(relation.node1.id, 'annotation 1')
            bioc_relation.add_node(node1)
            node2 = bioc.BioCNode(relation.node2.id, 'annotation 2')
            bioc_relation.add_node(node2)


        with open(os.path.join(output_dir, doc_name + ".bioc.xml"), 'w') as fp:
            bioc.dump(collection, fp)

