import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=int, default=4, help='1-pretrain, 2-train, 3-test')
parser.add_argument('-verbose', action='store_true', help='1-print debug logs')
parser.add_argument('-config', default='demo.config')
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-data_file', default='data.pkl')
parser.add_argument('-ner_dir', default='ner')
parser.add_argument('-re_dir', default='re')
parser.add_argument('-ner_iter', type=int, default=50)
parser.add_argument('-re_iter', type=int, default=50)
parser.add_argument('-use_gold_ner', action='store_true', default=False)
parser.add_argument('-self_adv', default='no', help='no, grad, label')
parser.add_argument('-lambd', type=float, default=0.05)
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-mutual_adv', default='no', help='no, grad, label')
parser.add_argument('-shared', default='no', help='no, hard, reg, soft')
parser.add_argument('-unk_ratio', type=float, default=0.03)
parser.add_argument('-reg_hp', type=float, default=0.001)
parser.add_argument('-hidden_num', type=int, default=1)

opt = parser.parse_args()

