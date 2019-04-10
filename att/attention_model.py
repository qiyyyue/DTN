import math


class att_model:
    def __init__(self, kg_embd = None, w2v = None, file_name = None):
        self.kg_embd = kg_embd
        self.w2v = w2v
        self.file_name = file_name
        self.sum_u = 0.0

    def build(self):
        rf = open(self.file_name, 'r', encoding='utf8')
        sum_u = 0.0
        for line in rf:
            text = list(line.split("\t"))
            tmp_u = 0.0
            for x in text:
                if x in self.w2v:
                    tmp_vec = self.w2v[x]
                    for tmp_x in tmp_vec:
                        tmp_u += tmp_x
            tmp_u = math.exp(tmp_u)
            sum_u += tmp_u
        self.sum_u = sum_u



    def cal_triple_att(self, triple_matrix):
        u = 0.0
        for word_vec in triple_matrix:
            for x in word_vec:
                u += x
        u = math.exp(u)
        return u/self.sum_u

    def cal_att_weight(self, triple_matrix, triple_matrix_list):
        u = 0.0
        for word_vec in triple_matrix:
            for x in word_vec:
                u += x
        u = math.exp(u)

        sum_u = 0.0
        for tmp_matrix in triple_matrix_list:
            tmp_u = 0
            for tmp_word_vec in tmp_matrix:
                for tmp_x in tmp_word_vec:
                    tmp_u += tmp_x
            tmp_u = math.exp(tmp_u)
            sum_u += tmp_u

        return u/sum_u
