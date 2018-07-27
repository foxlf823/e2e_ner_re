

import preprocess_cotype
from model.seqmodel import SeqModel, SeqModel1
import torch
import os
import feature_extractor
from tqdm import tqdm
import bioc
import ner
import relation_extraction
from data_structure import *
from classifymodel import *
from model.wordsequence import WordSequence
import logging
from utils.functions import normalize_word
import my_utils1

def pipeline(data, opt, test_file):
    test_token, test_entity, test_relation, test_name = preprocess_cotype.loadPreprocessData(test_file)

    data.load(opt.data_file)
    data.MAX_SENTENCE_LENGTH = -1
    data.show_data_summary()

    data.fix_alphabet()
    seq_model = SeqModel(data)
    seq_model.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'model.pkl')))
    seq_wordseq = WordSequence(data, False, True, True, data.use_char)
    seq_wordseq.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'wordseq.pkl')))

    classify_model = ClassifyModel(data)
    if torch.cuda.is_available():
        classify_model = classify_model.cuda(data.HP_gpu)
    classify_model.load_state_dict(torch.load(os.path.join(opt.re_dir, 'model.pkl')))
    classify_wordseq = WordSequence(data, True, False, True, False)
    classify_wordseq.load_state_dict(torch.load(os.path.join(opt.re_dir, 'wordseq.pkl')))

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

            ncrf_data = generateDataForOneDoc(doc_token, doc_entity)

            data.raw_texts, data.raw_Ids = ner.read_instanceFromBuffer(ncrf_data, data.word_alphabet, data.char_alphabet,
                                                         data.feature_alphabets, data.label_alphabet, data.number_normalized,
                                                         data.MAX_SENTENCE_LENGTH)


            decode_results = ner.evaluateWhenTest(data, seq_wordseq, seq_model)


            entities = translateNCRFPPintoEntities(doc_token, decode_results, doc_name)



        test_X, test_other = getRelationInstanceForOneDoc(doc_token, entities, doc_name, data)

        relations = evaluateWhenTest(classify_wordseq, classify_model, test_X, data, test_other, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']])

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
            if gold['type'] == 'None': # we don't count None relations
                continue
            gold_relation = Relation()
            gold_relation.type = gold['type']
            entity1 = doc_entity[(gold['entity1_id']==doc_entity['id'])].iloc[0]
            entity2 = doc_entity[(gold['entity2_id'] == doc_entity['id'])].iloc[0]
            node1 = Entity()
            node1.start = entity1['start']
            node1.end = entity1['end']
            node1.type = entity1['type']
            gold_relation.node1 = node1
            node2 = Entity()
            node2.start = entity2['start']
            node2.end = entity2['end']
            node2.type = entity2['type']
            gold_relation.node2 = node2

            gold_relations.append(gold_relation)


        total_re += len(gold_relations)

        for predict in relations:
            for gold in gold_relations:
                if predict.equals_cotype(gold):
                    correct_re += 1
                    break



    ner_p = correct_ner*1.0/predict_ner
    ner_r = correct_ner*1.0/total_ner
    ner_f1 = 2.0*ner_p*ner_r/(ner_p+ner_r)
    print("NER p: %.4f | r: %.4f | f1: %.4f" % (ner_p, ner_r, ner_f1))

    re_p = correct_re*1.0/predict_re
    re_r = correct_re*1.0/total_re
    re_f1 = 2.0*re_p*re_r/(re_p+re_r)
    print("RE p: %.4f | r: %.4f | f1: %.4f" % (re_p, re_r, re_f1))



def evaluateWhenTest(wordseq, model, instances, data, test_other, relationVocab):
    wordseq.eval()
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
            hidden = wordseq.forward(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                     batch_charrecover, position1_seq_tensor, position2_seq_tensor)
            pred = model.forward(hidden, batch_wordlen,
                                 e1_token, e1_length, e2_token, e2_length, e1_type, e2_type, tok_num_betw, et_num)

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

def getRelationInstanceForOneDoc(doc_token, entities, doc_name, data):
    X = []
    other = []

    row_num = len(entities)

    for latter_idx in range(row_num):

        for former_idx in range(row_num):

            if former_idx < latter_idx:

                former = entities[former_idx]
                latter = entities[latter_idx]

                context_token = doc_token
                former_tf_start, former_tf_end = former.tf_start, former.tf_end
                latter_tf_start, latter_tf_end = latter.tf_start, latter.tf_end

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
                        word = normalize_word(token['text'])
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

                features['e1_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(former.type)
                features['e2_type'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_TYPE]']].get_index(latter.type)
                features['e1_token'] = former_token
                features['e2_token'] = latter_token


                features['tok_num_betw'] = data.re_feature_alphabets[data.re_feature_name2id['[TOKEN_NUM]']].get_index(latter.tf_start-former.tf_end)

                entity_between = getEntitiesBetween(former, latter, entities)
                features['et_num'] = data.re_feature_alphabets[data.re_feature_name2id['[ENTITY_NUM]']].get_index(len(entity_between))

                X.append(features)

                other.append((former, latter))

    return X, other



def generateDataForOneDoc(doc_token, doc_entity):

    lines = []

    sent_token = doc_token
    sent_entity = doc_entity

    for _, token in sent_token.iterrows():
        word = token['text']
        pos = token['postag']
        cap = my_utils.featureCapital(word)
        label = preprocess_cotype.getLabel(token['start'], token['end'], sent_entity)

        lines.append("{} [Cap]{} [POS]{} {}\n".format(word, cap, pos, label))

    lines.append("\n")

    return lines

def translateNCRFPPintoEntities(doc_token, predict_results, doc_name):

    entity_id = 1
    results = []

    sent_length = len(predict_results[0][0])
    sent_token = doc_token
    sent_id = sent_token.iloc[0]['sent_idx']

    assert sent_token.shape[0] == sent_length, "file {}, sent {}".format(doc_name, 0)
    labelSequence = []

    for idy in range(sent_length):
        token = sent_token.iloc[idy]
        label = predict_results[0][0][idy]
        labelSequence.append(label)

        if label[0] == 'S' or label[0] == 'B':
            entity = Entity()
            entity.create(str(entity_id), label[2:], token['start'], token['end'], token['text'], sent_id, idy, idy)
            results.append(entity)
            entity_id += 1

        elif label[0] == 'M' or label[0] == 'E':
            if ner.checkWrongState(labelSequence):
                entity = results[-1]
                entity.append(token['start'], token['end'], token['text'], idy)


    return results

def getEntitiesBetween(former, latter, entities):
    results = []
    for entity in entities:
        if entity.start >= former.end and entity.end <= latter.start:
            results.append(entity)

    return results