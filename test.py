import preprocess
from model.seqmodel import SeqModel
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
from options import opt


def test1(data, opt, predict_dir):
    test_token, test_entity, test_relation, test_name = preprocess.loadPreprocessData(data.test_dir)

    # evaluate on test data and output results in bioc format, one doc one file

    data.load(opt.data_file)
    data.MAX_SENTENCE_LENGTH = -1
    data.show_data_summary()

    data.fix_alphabet()
    seq_model = SeqModel(data)
    seq_model.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'model.pkl')))
    seq_wordseq = WordSequence(data, False, True, data.use_char)
    seq_wordseq.load_state_dict(torch.load(os.path.join(opt.ner_dir, 'wordseq.pkl')))

    classify_model = ClassifyModel(data)
    if torch.cuda.is_available():
        classify_model = classify_model.cuda(data.HP_gpu)
    classify_model.load_state_dict(torch.load(os.path.join(opt.re_dir, 'model.pkl')))
    classify_wordseq = WordSequence(data, True, False, False)
    classify_wordseq.load_state_dict(torch.load(os.path.join(opt.re_dir, 'wordseq.pkl')))

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


            decode_results = ner.evaluateWhenTest(data, seq_wordseq, seq_model)


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

        relations = relation_extraction.evaluateWhenTest1(classify_wordseq, classify_model, test_X, data, test_other, data.re_feature_alphabets[data.re_feature_name2id['[RELATION]']])

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