import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=int, default=4, help='1-preprocess, 2-pretrain, 3-train, other-test')
parser.add_argument('-verbose', action='store_true', help='1-print debug logs')
parser.add_argument('-config', default='demo.train.config')
parser.add_argument('-random_seed', type=int, default=1)

opt = parser.parse_args()

