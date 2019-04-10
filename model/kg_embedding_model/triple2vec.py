from numpy import *
import json

class triple2vec:

    entity2id_dict = dict()
    relation2id_dict = dict()

    relationes_dict = []
    entities_dict = []

    def __init__(self, dim):
        #load model
        dir = "../model/kg_embedding_model/TML1K/"
        model = open(dir + "embedding.vec.json")
        model_vec = json.load(model)

        self.relationes_dict = model_vec["rel_embeddings"]
        self.entities_dict = model_vec["ent_embeddings"]
        self.dim = dim

        #load dict
        with open(dir + "entity2id.txt") as file:
            lines = file.readlines()
            for line in lines:
                if len(line.strip().split("\t")) < 2:
                    continue
                entity = line.strip().split("\t")[0]
                id = line.strip().split("\t")[1]
                self.entity2id_dict[entity] = int(id)

        with open(dir + "relation2id.txt") as file:
            lines = file.readlines()
            for line in lines:
                if len(line.strip().split("\t")) < 2:
                    continue
                relation = line.strip().split("\t")[0]
                id = line.strip().split("\t")[1]
                self.relation2id_dict[relation] = int(id)

    def getNumofCommonSubstr(self, str1, str2):
        lstr1 = len(str1)
        lstr2 = len(str2)
        record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
        maxNum = 0
        p = 0

        for i in range(lstr1):
            for j in range(lstr2):
                if str1[i] == str2[j]:
                    record[i + 1][j + 1] = record[i][j] + 1
                    if record[i + 1][j + 1] > maxNum:
                        maxNum = record[i + 1][j + 1]
                        p = i + 1
        return maxNum

    def entity2vec(self, entity):
        entity_id = -1

        if entity in self.entity2id_dict.keys():
            entity_id = self.entity2id_dict.get(entity)
        else:
            for entity_str in self.entity2id_dict.keys():
                if entity in entity_str:
                    entity_id = self.entity2id_dict.get(entity_str)
                    break
                elif entity_str in entity:
                    entity_id = self.entity2id_dict.get(entity_str)
                    break
                elif self.getNumofCommonSubstr(entity, entity_str) > 3:
                    entity_id = self.entity2id_dict.get(entity_str)
                    break

        if entity_id == -1:
            return [0]*self.dim
        else:
            return self.entities_dict[entity_id]
    #
    def relation2vec(self, relation):
        relation_id = -1

        if relation in self.relation2id_dict.keys():
            relation_id = self.relation2id_dict.get(relation)
        else:
            for realation_str in self.relation2id_dict.keys():
                if relation in realation_str:
                    relation_id = self.relation2id_dict.get(realation_str)
                    break
                elif realation_str in relation:
                    relation_id = self.relation2id_dict.get(realation_str)
                    break
                elif self.getNumofCommonSubstr(relation, realation_str) > 3:
                    relation_id = self.relation2id_dict.get(realation_str)
                    break
        if relation_id == -1:
            return [0]*self.dim
        else:
            return self.relationes_dict[relation_id]
    #
    def triple2vec(self, h, t, r):
        h_vec = self.entity2vec(h)
        t_vec = self.entity2vec(t)
        r_vec = self.relation2vec(r)
        if h_vec == None or t_vec == None or r_vec == None:
            return None
        return {"h": h_vec, "t": t_vec, "r": r_vec}






















