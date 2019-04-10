import random

from gensim.models import Word2Vec
from att.attention_model import att_model

file_name = "../data/train_news/2016_election/triples/fake/fake_4_triple.txt"
w2v = Word2Vec.load("../model/word2vec_model/2016_election/w2v_model.model")

att = att_model(w2v=w2v, file_name=file_name)
att.build()

print(att.sum_u)

rf = open(file_name, 'r')
for line in rf:
    text = list(line.split("\t"))
    text_vec = [w2v[x] for x in text if x in w2v]
    print(att.cal_triple_att(text_vec)*(random.uniform(0.5, 1.5)))