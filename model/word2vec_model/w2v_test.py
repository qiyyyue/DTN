from gensim.models import Word2Vec

from data.cnews_loader import word_to_vec

word_to_vec = Word2Vec.load("w2v_model.model")

print(type(word_to_vec['people']), word_to_vec['people'])