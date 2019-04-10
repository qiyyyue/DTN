import numpy

from model.kg_embedding_model.triple2vec import triple2vec

t2v = triple2vec(20)

print(t2v.entity2vec("Donald trump"))
print(type(numpy.array(t2v.entity2vec("Donald trump"))), numpy.array(t2v.entity2vec("Donald trump")))