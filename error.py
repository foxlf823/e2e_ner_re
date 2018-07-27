import preprocess
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
from utils.data import data
from options import opt


def error_pipeline(data, opt):
    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(data.test_dir)

    # evaluate on test data and output results in bioc format, one doc one file

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

    error_dir = "error"
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    for i in tqdm(range(len(test_name))):
        doc_name = test_name[i]
        doc_token = test_token[i]
        doc_entity = test_entity[i]
        doc_relation = test_relation[i]

        listEntityFP = []
        listEntityFN = []

        ncrf_data = ner.generateDataForOneDoc(doc_token, doc_entity)

        data.raw_texts, data.raw_Ids = ner.read_instanceFromBuffer(ncrf_data, data.word_alphabet, data.char_alphabet,
                                                     data.feature_alphabets, data.label_alphabet, data.number_normalized,
                                                     data.MAX_SENTENCE_LENGTH)


        decode_results = ner.evaluateWhenTest(data, seq_wordseq, seq_model)


        entities = ner.translateNCRFPPintoEntities(doc_token, decode_results, doc_name)

        # entity fn
        for _, gold in doc_entity.iterrows():
            find = False
            for predict in entities:
                if gold['type'] == predict.type and gold['start'] == predict.start and gold['end'] == predict.end:
                    find = True
                    break
            if not find:
                context_token = doc_token[(doc_token['sent_idx'] == gold['sent_idx']) ]
                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == gold['start']:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == gold['end'] :
                        sequence += "]"
                    sequence += " "
                listEntityFN.append(
                    "{} | {}\n{}\n".format(gold['text'], gold['type'],sequence))

        # entity fp
        for predict in entities:
            find = False
            for _, gold in doc_entity.iterrows():
                if gold['type'] == predict.type and gold['start'] == predict.start and gold['end'] == predict.end:
                    find = True
                    break
            if not find:
                context_token = doc_token[(doc_token['sent_idx'] == predict.sent_idx) ]
                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == predict.start:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == predict.end :
                        sequence += "]"
                    sequence += " "
                listEntityFP.append(
                    "{} | {}\n{}\n".format(predict.text, predict.type,sequence))


        test_X, test_other = relation_extraction.getRelationInstanceForOneDoc(doc_token, entities, doc_name, data)

        relations = relation_extraction.evaluateWhenTest(classify_wordseq, classify_model, test_X, data, test_other, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']])

        listRelationFP = []
        listRelationFN = []
        # relation fn
        for _, gold in doc_relation.iterrows():
            find = False
            gold_entity1 = doc_entity[(doc_entity['id'] == gold['entity1_id'])].iloc[0]
            gold_entity2 = doc_entity[(doc_entity['id'] == gold['entity2_id'])].iloc[0]

            for predict in relations:
                predict_entity1 = predict.node1
                predict_entity2 = predict.node2

                if gold['type'] == predict.type \
                    and gold_entity1['type']==predict_entity1.type and gold_entity1['start']==predict_entity1.start and gold_entity1['end']==predict_entity1.end \
                    and gold_entity2['type']==predict_entity2.type and gold_entity2['start']==predict_entity2.start and gold_entity2['end']==predict_entity2.end:
                    find = True
                    break
                elif gold['type'] == predict.type \
                    and gold_entity1['type']==predict_entity2.type and gold_entity1['start']==predict_entity2.start and gold_entity1['end']==predict_entity2.end \
                    and gold_entity2['type']==predict_entity1.type and gold_entity2['start']==predict_entity1.start and gold_entity2['end']==predict_entity1.end:
                    find = True
                    break



            if not find:
                former = gold_entity1 if gold_entity1['start'] < gold_entity2['start'] else gold_entity2
                latter = gold_entity2 if gold_entity1['start'] < gold_entity2['start'] else gold_entity1
                context_token = doc_token[
                    (doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]

                # print("{}: {} | {}: {}".format(former['id'], former['text'], latter['id'], latter['text']))
                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == former['start'] or token['start'] == latter['start']:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == former['end'] or token['end'] == latter['end']:
                        sequence += "]"
                    sequence += " "

                listRelationFN.append(
                    "{} | {} | {}\n{}\n".format(former['text'], latter['text'], gold['type'], sequence))

        # relation fp
        for predict in relations:
            predict_entity1 = predict.node1
            predict_entity2 = predict.node2
            find = False

            for _, gold in doc_relation.iterrows():

                gold_entity1 = doc_entity[(doc_entity['id'] == gold['entity1_id'])].iloc[0]
                gold_entity2 = doc_entity[(doc_entity['id'] == gold['entity2_id'])].iloc[0]

                if gold['type'] == predict.type \
                    and gold_entity1['type']==predict_entity1.type and gold_entity1['start']==predict_entity1.start and gold_entity1['end']==predict_entity1.end \
                    and gold_entity2['type']==predict_entity2.type and gold_entity2['start']==predict_entity2.start and gold_entity2['end']==predict_entity2.end:
                    find = True
                    break
                elif gold['type'] == predict.type \
                    and gold_entity1['type']==predict_entity2.type and gold_entity1['start']==predict_entity2.start and gold_entity1['end']==predict_entity2.end \
                    and gold_entity2['type']==predict_entity1.type and gold_entity2['start']==predict_entity1.start and gold_entity2['end']==predict_entity1.end:
                    find = True
                    break


            if not find:
                former = predict_entity1 if predict_entity1.start < predict_entity2.start else predict_entity2
                latter = predict_entity2 if predict_entity1.start < predict_entity2.start else predict_entity1
                context_token = doc_token[
                    (doc_token['sent_idx'] >= former.sent_idx) & (doc_token['sent_idx'] <= latter.sent_idx)]

                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == former.start or token['start'] == latter.start:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == former.end or token['end'] == latter.end:
                        sequence += "]"
                    sequence += " "

                listRelationFP.append(
                    "{} | {} | {}\n{}\n".format(former.text, latter.text, predict.type, sequence))




        with open(os.path.join(error_dir, doc_name + ".txt"), 'w') as fp:
            fp.write("######## ENTITY FN ERROR ##########\n\n")
            for item in listEntityFN:
                fp.write(item)
                fp.write('\n')

            fp.write("######## ENTITY FP ERROR ##########\n\n")
            for item in listEntityFP:
                fp.write(item)
                fp.write('\n')

            fp.write("######## RELATION FN ERROR ##########\n\n")
            for item in listRelationFN:
                fp.write(item)
                fp.write('\n')

            fp.write("######## RELATION FP ERROR ##########\n\n")
            for item in listRelationFP:
                fp.write(item)
                fp.write('\n')


def error_sp(data, opt): # shared-private
    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(data.test_dir)

    # evaluate on test data and output results in bioc format, one doc one file

    data.load(opt.data_file)
    data.MAX_SENTENCE_LENGTH = -1
    data.show_data_summary()

    data.fix_alphabet()
    seq_model = SeqModel1(data)
    seq_model.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'model.pkl')))
    seq_wordseq = WordSequence(data, False, True, True, data.use_char)
    seq_wordseq.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'wordseq.pkl')))

    classify_model = ClassifyModel1(data)
    classify_model.load_state_dict(torch.load(os.path.join(opt.re_dir, 'model.pkl')))
    classify_wordseq = WordSequence(data, True, False, True, False)
    classify_wordseq.load_state_dict(torch.load(os.path.join(opt.re_dir, 'wordseq.pkl')))

    wordseq_shared = WordSequence(data, False, False, False, False)
    wordseq_shared.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'wordseq_shared.pkl')))

    error_dir = "error"
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    for i in tqdm(range(len(test_name))):
        doc_name = test_name[i]
        doc_token = test_token[i]
        doc_entity = test_entity[i]
        doc_relation = test_relation[i]

        listEntityFP = []
        listEntityFN = []

        ncrf_data = ner.generateDataForOneDoc(doc_token, doc_entity)

        data.raw_texts, data.raw_Ids = ner.read_instanceFromBuffer(ncrf_data, data.word_alphabet, data.char_alphabet,
                                                     data.feature_alphabets, data.label_alphabet, data.number_normalized,
                                                     data.MAX_SENTENCE_LENGTH)


        decode_results = ner.evaluateWhenTest1(data, seq_wordseq, wordseq_shared, seq_model)


        entities = ner.translateNCRFPPintoEntities(doc_token, decode_results, doc_name)

        # entity fn
        for _, gold in doc_entity.iterrows():
            find = False
            for predict in entities:
                if gold['type'] == predict.type and gold['start'] == predict.start and gold['end'] == predict.end:
                    find = True
                    break
            if not find:
                context_token = doc_token[(doc_token['sent_idx'] == gold['sent_idx']) ]
                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == gold['start']:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == gold['end'] :
                        sequence += "]"
                    sequence += " "
                listEntityFN.append(
                    "{} | {}\n{}\n".format(gold['text'], gold['type'],sequence))

        # entity fp
        for predict in entities:
            find = False
            for _, gold in doc_entity.iterrows():
                if gold['type'] == predict.type and gold['start'] == predict.start and gold['end'] == predict.end:
                    find = True
                    break
            if not find:
                context_token = doc_token[(doc_token['sent_idx'] == predict.sent_idx) ]
                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == predict.start:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == predict.end :
                        sequence += "]"
                    sequence += " "
                listEntityFP.append(
                    "{} | {}\n{}\n".format(predict.text, predict.type,sequence))



        test_X, test_other = relation_extraction.getRelationInstanceForOneDoc(doc_token, entities, doc_name, data)

        relations = relation_extraction.evaluateWhenTest1(classify_wordseq, wordseq_shared, classify_model, test_X, data, test_other, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']])

        listRelationFP = []
        listRelationFN = []
        # relation fn
        for _, gold in doc_relation.iterrows():
            find = False
            gold_entity1 = doc_entity[(doc_entity['id'] == gold['entity1_id'])].iloc[0]
            gold_entity2 = doc_entity[(doc_entity['id'] == gold['entity2_id'])].iloc[0]

            for predict in relations:
                predict_entity1 = predict.node1
                predict_entity2 = predict.node2

                if gold['type'] == predict.type \
                    and gold_entity1['type']==predict_entity1.type and gold_entity1['start']==predict_entity1.start and gold_entity1['end']==predict_entity1.end \
                    and gold_entity2['type']==predict_entity2.type and gold_entity2['start']==predict_entity2.start and gold_entity2['end']==predict_entity2.end:
                    find = True
                    break
                elif gold['type'] == predict.type \
                    and gold_entity1['type']==predict_entity2.type and gold_entity1['start']==predict_entity2.start and gold_entity1['end']==predict_entity2.end \
                    and gold_entity2['type']==predict_entity1.type and gold_entity2['start']==predict_entity1.start and gold_entity2['end']==predict_entity1.end:
                    find = True
                    break



            if not find:
                former = gold_entity1 if gold_entity1['start'] < gold_entity2['start'] else gold_entity2
                latter = gold_entity2 if gold_entity1['start'] < gold_entity2['start'] else gold_entity1
                context_token = doc_token[
                    (doc_token['sent_idx'] >= former['sent_idx']) & (doc_token['sent_idx'] <= latter['sent_idx'])]

                # print("{}: {} | {}: {}".format(former['id'], former['text'], latter['id'], latter['text']))
                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == former['start'] or token['start'] == latter['start']:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == former['end'] or token['end'] == latter['end']:
                        sequence += "]"
                    sequence += " "

                listRelationFN.append(
                    "{} | {} | {}\n{}\n".format(former['text'], latter['text'], gold['type'], sequence))

        # relation fp
        for predict in relations:
            predict_entity1 = predict.node1
            predict_entity2 = predict.node2
            find = False

            for _, gold in doc_relation.iterrows():

                gold_entity1 = doc_entity[(doc_entity['id'] == gold['entity1_id'])].iloc[0]
                gold_entity2 = doc_entity[(doc_entity['id'] == gold['entity2_id'])].iloc[0]

                if gold['type'] == predict.type \
                    and gold_entity1['type']==predict_entity1.type and gold_entity1['start']==predict_entity1.start and gold_entity1['end']==predict_entity1.end \
                    and gold_entity2['type']==predict_entity2.type and gold_entity2['start']==predict_entity2.start and gold_entity2['end']==predict_entity2.end:
                    find = True
                    break
                elif gold['type'] == predict.type \
                    and gold_entity1['type']==predict_entity2.type and gold_entity1['start']==predict_entity2.start and gold_entity1['end']==predict_entity2.end \
                    and gold_entity2['type']==predict_entity1.type and gold_entity2['start']==predict_entity1.start and gold_entity2['end']==predict_entity1.end:
                    find = True
                    break


            if not find:
                former = predict_entity1 if predict_entity1.start < predict_entity2.start else predict_entity2
                latter = predict_entity2 if predict_entity1.start < predict_entity2.start else predict_entity1
                context_token = doc_token[
                    (doc_token['sent_idx'] >= former.sent_idx) & (doc_token['sent_idx'] <= latter.sent_idx)]

                sequence = ""
                for _, token in context_token.iterrows():
                    if token['start'] == former.start or token['start'] == latter.start:
                        sequence += "["
                    sequence += token['text']
                    if token['end'] == former.end or token['end'] == latter.end:
                        sequence += "]"
                    sequence += " "

                listRelationFP.append(
                    "{} | {} | {}\n{}\n".format(former.text, latter.text, predict.type, sequence))




        with open(os.path.join(error_dir, doc_name + ".txt"), 'w') as fp:
            fp.write("######## ENTITY FN ERROR ##########\n\n")
            for item in listEntityFN:
                fp.write(item)
                fp.write('\n')

            fp.write("######## ENTITY FP ERROR ##########\n\n")
            for item in listEntityFP:
                fp.write(item)
                fp.write('\n')

            fp.write("######## RELATION FN ERROR ##########\n\n")
            for item in listRelationFN:
                fp.write(item)
                fp.write('\n')

            fp.write("######## RELATION FP ERROR ##########\n\n")
            for item in listRelationFP:
                fp.write(item)
                fp.write('\n')



# error_pipeline(data, opt)

error_sp(data, opt)