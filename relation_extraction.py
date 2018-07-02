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


import vocab
import my_utils
from feature_extractor import *
from utils.data import data
from data_structure import *

def dataset_stat(tokens, entities, relations):
    word_alphabet = sortedcontainers.SortedSet()
    postag_alphabet = sortedcontainers.SortedSet()
    relation_alphabet = sortedcontainers.SortedSet()
    entity_type_alphabet = sortedcontainers.SortedSet()
    entity_alphabet = sortedcontainers.SortedSet()

    for i, doc_token in enumerate(tokens):

        doc_entity = entities[i]
        doc_relation = relations[i]

        sent_idx = 0
        sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]
        while sentence.shape[0] != 0:
            for _, token in sentence.iterrows():
                word_alphabet.add(my_utils.normalizeWord(token['text']))
                postag_alphabet.add(token['postag'])

            entities_in_sentence = doc_entity[(doc_entity['sent_idx'] == sent_idx)]
            for _, entity in entities_in_sentence.iterrows():
                entity_type_alphabet.add(entity['type'])
                tk_idx = entity['tf_start']
                while tk_idx <= entity['tf_end']:
                    entity_alphabet.add(my_utils.normalizeWord(sentence.iloc[tk_idx, 0])) # assume 'text' is in 0 column
                    tk_idx += 1

            sent_idx += 1
            sentence = doc_token[(doc_token['sent_idx'] == sent_idx)]

        for _, relation in doc_relation.iterrows():
            relation_alphabet.add(relation['type'])

    return word_alphabet, postag_alphabet, relation_alphabet, entity_type_alphabet, entity_alphabet

def pretrain(train_token, train_entity, train_relation, train_name, test_token, test_entity, test_relation, test_name,
             data):
    word_alphabet, postag_alphabet, relation_alphabet, entity_type_alphabet, entity_alphabet = dataset_stat(train_token, train_entity, train_relation)
    logging.info("training dataset stat completed")
    if data.full_data:
        test_word_alphabet, test_postag_alphabet, test_relation_alphabet, test_entity_type_alphabet, test_entity_alphabet = dataset_stat(test_token, test_entity, test_relation)
        word_alphabet = word_alphabet | test_word_alphabet
        postag_alphabet = postag_alphabet | test_postag_alphabet
        relation_alphabet = relation_alphabet | test_relation_alphabet
        entity_type_alphabet = entity_type_alphabet | test_entity_type_alphabet
        entity_alphabet = entity_alphabet | test_entity_alphabet
        del test_word_alphabet, test_postag_alphabet, test_relation_alphabet, test_entity_type_alphabet, test_entity_alphabet
        logging.info("test dataset stat completed")

    position_alphabet = sortedcontainers.SortedSet()
    for i in range(data.max_seq_len):
        position_alphabet.add(i)
        position_alphabet.add(-i)

    relation_vocab = vocab.Vocab(relation_alphabet, None, data.feat_config['[RELATION]']['emb_size'], data)
    word_vocab = vocab.Vocab(word_alphabet, data.word_emb_dir, data.word_emb_dim, data)
    postag_vocab = vocab.Vocab(postag_alphabet, None, data.feat_config['[POS]']['emb_size'], data)
    entity_type_vocab = vocab.Vocab(entity_type_alphabet, None, data.feat_config['[ENTITY_TYPE]']['emb_size'], data)
    entity_vocab = vocab.Vocab(entity_alphabet, None, data.feat_config['[ENTITY]']['emb_size'], data)
    position_vocab1 = vocab.Vocab(position_alphabet, None, data.feat_config['[POSITION]']['emb_size'], data)
    position_vocab2 = vocab.Vocab(position_alphabet, None, data.feat_config['[POSITION]']['emb_size'], data)
    # we directly use position_alphabet to build them, since they are all numbers
    tok_num_betw_vocab = vocab.Vocab(position_alphabet, None, data.feat_config['[POSITION]']['emb_size'], data)
    et_num_vocab = vocab.Vocab(position_alphabet, None, data.feat_config['[POSITION]']['emb_size'], data)
    logging.info("vocab build completed")

    logging.info("saving ... vocab")
    pickle.dump(word_vocab, open(os.path.join(data.pretrain, 'word_vocab.pkl'), "wb"), True)
    pickle.dump(postag_vocab, open(os.path.join(data.pretrain, 'postag_vocab.pkl'), "wb"), True)
    pickle.dump(relation_vocab, open(os.path.join(data.pretrain, 'relation_vocab.pkl'), "wb"), True)
    pickle.dump(entity_type_vocab, open(os.path.join(data.pretrain, 'entity_type_vocab.pkl'), "wb"), True)
    pickle.dump(entity_vocab, open(os.path.join(data.pretrain, 'entity_vocab.pkl'), "wb"), True)
    pickle.dump(position_vocab1, open(os.path.join(data.pretrain, 'position_vocab1.pkl'), "wb"), True)
    pickle.dump(position_vocab2, open(os.path.join(data.pretrain, 'position_vocab2.pkl'), "wb"), True)
    pickle.dump(tok_num_betw_vocab, open(os.path.join(data.pretrain, 'tok_num_betw_vocab.pkl'), "wb"), True)
    pickle.dump(et_num_vocab, open(os.path.join(data.pretrain, 'et_num_vocab.pkl'), "wb"), True)

    train_X, train_Y, _ = my_utils.getRelationInstance2(train_token, train_entity, train_relation, train_name, word_vocab, postag_vocab,
                                                     relation_vocab, entity_type_vocab,
                                                     entity_vocab, position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab)
    logging.info("training instance build completed, total {}".format(len(train_Y)))
    pickle.dump(train_X, open(os.path.join(data.pretrain, 'train_X.pkl'), "wb"), True)
    pickle.dump(train_Y, open(os.path.join(data.pretrain, 'train_Y.pkl'), "wb"), True)


    test_X, test_Y, test_other = my_utils.getRelationInstance2(test_token, test_entity, test_relation, test_name, word_vocab, postag_vocab,
                                                            relation_vocab, entity_type_vocab,
                                                            entity_vocab, position_vocab1, position_vocab2, tok_num_betw_vocab, et_num_vocab)
    logging.info("test instance build completed, total {}".format(len(test_Y)))
    pickle.dump(test_X, open(os.path.join(data.pretrain, 'test_X.pkl'), "wb"), True)
    pickle.dump(test_Y, open(os.path.join(data.pretrain, 'test_Y.pkl'), "wb"), True)
    pickle.dump(test_other, open(os.path.join(data.pretrain, 'test_Other.pkl'), "wb"), True)

def makeDatasetWithoutUnknown(test_X, test_Y, relation_vocab, b_shuffle, my_collate):
    test_X_remove_unk = []
    test_Y_remove_unk = []
    for i in range(len(test_X)):
        x = test_X[i]
        y = test_Y[i]

        if y != relation_vocab.unk_idx:
            test_X_remove_unk.append(x)
            test_Y_remove_unk.append(y)

    test_set = my_utils.RelationDataset(test_X_remove_unk, test_Y_remove_unk)
    test_loader = DataLoader(test_set, data.HP_batch_size, shuffle=b_shuffle, collate_fn=my_collate)
    it = iter(test_loader)
    logging.info("instance after removing unknown, {}".format(len(test_Y_remove_unk)))
    return test_loader, it

def randomSampler(dataset_list, ratio):
    a = range(len(dataset_list))
    random.shuffle(a)
    indices = a[:int(len(dataset_list)*ratio)]
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    return sampler

def makeDatasetUnknown(test_X, test_Y, relation_vocab, my_collate, ratio):
    test_X_remove_unk = []
    test_Y_remove_unk = []
    for i in range(len(test_X)):
        x = test_X[i]
        y = test_Y[i]

        if y == relation_vocab.unk_idx:
            test_X_remove_unk.append(x)
            test_Y_remove_unk.append(y)

    test_set = my_utils.RelationDataset(test_X_remove_unk, test_Y_remove_unk)

    test_loader = DataLoader(test_set, data.HP_batch_size, shuffle=False, sampler=randomSampler(test_Y_remove_unk, ratio), collate_fn=my_collate)
    it = iter(test_loader)

    return test_loader, it

def train():

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

    # One relation instance is composed of X (a pair of entities and their context), Y (relation label).
    train_X = pickle.load(open(os.path.join(data.pretrain, 'train_X.pkl'), 'rb'))
    train_Y = pickle.load(open(os.path.join(data.pretrain, 'train_Y.pkl'), 'rb'))

    logging.info("total training instance {}".format(len(train_Y)))

    # cnnrnn
    # my_collate = my_utils.unsorted_collate
    my_collate = my_utils.sorted_collate

    train_loader, train_iter = makeDatasetWithoutUnknown(train_X, train_Y, relation_vocab, True, my_collate)
    num_iter = len(train_loader)
    unk_loader, unk_iter = makeDatasetUnknown(train_X, train_Y, relation_vocab, my_collate, data.unk_ratio)


    test_X = pickle.load(open(os.path.join(data.pretrain, 'test_X.pkl'), 'rb'))
    test_Y = pickle.load(open(os.path.join(data.pretrain, 'test_Y.pkl'), 'rb'))
    test_Other = pickle.load(open(os.path.join(data.pretrain, 'test_Other.pkl'), 'rb'))
    logging.info("total test instance {}".format(len(test_Y)))
    test_loader = DataLoader(my_utils.RelationDataset(test_X, test_Y),
                              data.HP_batch_size, shuffle=False, collate_fn=my_collate) # drop_last=True


    # cnnrnn
    m_low = LSTMFeatureExtractor(word_vocab, postag_vocab, position_vocab1, position_vocab2,
                                                 1, data.seq_feature_size, data.HP_dropout)
    # m_low = CNNFeatureExtractor(word_vocab, postag_vocab, position_vocab1, position_vocab2,
    #                                         1, data.seq_feature_size,
    #                                         3, [3,4,5], data.HP_dropout)

    if torch.cuda.is_available():
        m_low = m_low.cuda(data.gpu)


    m = MLP(data.seq_feature_size, relation_vocab, entity_type_vocab, entity_vocab, tok_num_betw_vocab,
                                     et_num_vocab)


    if torch.cuda.is_available():
        m = m.cuda(data.gpu)

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


        unk_loader, unk_iter = makeDatasetUnknown(train_X, train_Y, relation_vocab, my_collate, data.unk_ratio)

        logging.info('epoch {} end'.format(epoch))
        logging.info('Train Accuracy: {}%'.format(100.0 * correct / total))

        test_accuracy = evaluate(m_low, m, test_loader, test_Other)
        # test_accuracy = evaluate(m, test_loader)
        logging.info('Test Accuracy: {}%'.format(test_accuracy))

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(m_low.state_dict(), '{}/feature_extractor.pth'.format(data.output))
            torch.save(m.state_dict(), '{}/model.pth'.format(data.output))
            pickle.dump(test_Other, open(os.path.join(data.output, 'results.pkl'), "wb"), True)
            logging.info('New best accuracy: {}'.format(best_acc))


    logging.info("training completed")


def evaluate(feature_extractor, m, loader, other):
    #results = []
    feature_extractor.eval()
    m.eval()
    it = iter(loader)
    start, end = 0, 0
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

        start = end
        end = end + targets.size(0)

        # we use sorted_collate, so we need to unsorted them during evaluate
        # cnnrnn
        _, unsort_idx = sort_idx.sort(0, descending=False)
        pred = pred.index_select(0, unsort_idx)

        for i, d in enumerate(other[start:end]):
            d["type"] = pred[i].item()

        # iii += 1

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
        x2, x1, _ = my_utils.my_collate(instance, True)
        # x2, x1, _ = my_utils.my_collate(instance, False)

        with torch.no_grad():

            _, _, _, _, _, _, _, sort_idx = x1

            hidden_features = feature_extractor.forward(x2, x1)

            outputs = m.forward(hidden_features, x2, x1)

            _, pred = torch.max(outputs, 1)


        # we use sorted_collate, so we need to unsorted them during evaluate
        # cnnrnn
        _, unsort_idx = sort_idx.sort(0, descending=False)
        pred = pred.index_select(0, unsort_idx)



        for i in range(start,end):

            former = test_other[i][0]
            latter = test_other[i][1]

            relation_type = relationVocab.lookup_id2str(pred[i-start].item())
            if relation_type == '<unk>':
                continue
            elif my_utils.relationConstraint1(relation_type, former.type, latter.type) == False:
                continue
            else:
                relation = Relation()
                relation.create(str(relation_id), relation_type, former, latter)
                relations.append(relation)

                relation_id += 1

    return relations



def test2(test_token, test_entity, test_relation, test_name, result_dumpdir):
    logging.info("loading ... vocab")
    relation_vocab = pickle.load(open(os.path.join(data.pretrain, 'relation_vocab.pkl'), 'rb'))

    logging.info("loading ... result")
    results = pickle.load(open(os.path.join(data.output, 'results.pkl'), "rb"))

    for i in tqdm(range(len(test_relation))):

        doc_entity = test_entity[i]
        doc_name = test_name[i]

        collection = bioc.BioCCollection()
        document = bioc.BioCDocument()
        collection.add_document(document)
        document.id = doc_name
        passage = bioc.BioCPassage()
        document.add_passage(passage)
        passage.offset = 0

        for _, entity in doc_entity.iterrows():
            anno_entity = bioc.BioCAnnotation()
            passage.add_annotation(anno_entity)
            anno_entity.id = entity['id']
            anno_entity.infons['type'] = entity['type']
            anno_entity_location = bioc.BioCLocation(entity['start'], entity['end'] - entity['start'])
            anno_entity.add_location(anno_entity_location)
            anno_entity.text = entity['text']

        relation_id = 1
        for result in results:

            if doc_name == result['doc_name'] :

                former = doc_entity[ (doc_entity['id'] == result['former_id'])].iloc[0]
                latter = doc_entity[(doc_entity['id'] == result['latter_id'])].iloc[0]

                relation_type = relation_vocab.lookup_id2str(result['type'])
                if relation_type == '<unk>':
                    continue
                elif my_utils.relationConstraint1(relation_type, former['type'], latter['type']) == False:
                    continue
                else:
                    bioc_relation = bioc.BioCRelation()
                    passage.add_relation(bioc_relation)
                    bioc_relation.id = str(relation_id)
                    relation_id += 1
                    bioc_relation.infons['type'] = relation_type

                    node1 = bioc.BioCNode(former['id'], 'annotation 1')
                    bioc_relation.add_node(node1)
                    node2 = bioc.BioCNode(latter['id'], 'annotation 2')
                    bioc_relation.add_node(node2)

        with open(os.path.join(result_dumpdir, doc_name + ".bioc.xml"), 'w') as fp:
            bioc.dump(collection, fp)