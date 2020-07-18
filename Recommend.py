import os
import pickle
from itertools import islice
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

class Recommend():

    def __init__(self, user_item_mat, vector_dict):

        self.user_item_mat = user_item_mat
        self.user_vectors = vector_dict['user_vec'] # user latent vector
        self.item_vectors = vector_dict['item_vec'] # item latent vector
        self.user_biases = vector_dict['user_bias']
        self.item_biases = vector_dict['item_bias']

        self.num_users = vector_dict['info']['user_num']
        self.num_items = vector_dict['info']['item_num']

    def get_item_like_probability(self):
        """
        user가 각 item을 좋아할 확률을 구하는 함수
        """
        ones = np.ones((self.num_users, self.num_items))
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        A = np.exp(A)
        A /= (A + ones)
        A = np.round(A, 3)

        self.item_like_prob_mat = A

        return A

    def get_similar_items(self, N = 30, item_ids = None):
        """
        각 아이템과 비슷한 아이템을 뽑아주는 함수
        """
        normed_factors = normalize(self.item_vectors)
        knn = NearestNeighbors(n_neighbors = N+1, metric = 'euclidean')
        knn.fit(normed_factors)

        if item_ids is not None:
            normed_factors = normed_factors[item_ids]
        _, items = knn.kneighbors(normed_factors)
        similar_items = items[:, 1:].astype(np.uint32) #자기 자신 제외

        return similar_items

    def get_popularity_rank(self, recommend_arr):
        """
        인기아이템 랭킹 계산 함수
        return: [1위를 가장 많이한 아이템, 2위를 가장 많이한 아이템, ...]
        """
        item_cnt_dic_lst = []
        for rec_lst in recommend_arr.T:
            item_cnt_dic = Counter(rec_lst)
            item_cnt_dic_lst.append(item_cnt_dic)

        popularity_rank = []
        for item_cnt_dic in item_cnt_dic_lst:
            ranked_item_cnt_dic = dict(sorted(item_cnt_dic.items(), key=lambda x:x[1], reverse=True))
            i=0
            while True:
                item = list(ranked_item_cnt_dic.keys())[i]
                if item not in popularity_rank:
                    popularity_rank.append(item)
                    break
                else:
                    i=i+1

        return popularity_rank

    def recommend(self, n):
        """
        추천 결과를 뽑아주는 함수
        """
        item_like_prob_mat = self.get_item_like_probability()

        recommend_lst = []
        for user_data, prob_lst in zip(self.user_item_mat, item_like_prob_mat):

            liked = set(np.nonzero(user_data)[0])
            count = n + len(liked)

            # 추천 item 리스트(선택하지 않은 item 추천)
            ids = np.argpartition(prob_lst, -count)[-count:]
            best_ids = np.argsort(prob_lst[ids])[::-1]
            best = ids[best_ids]
            top_n = list(islice((rec for rec in best if rec not in liked), n))
            recommend_lst.append(top_n)

        recommend_arr = np.array(recommend_lst)
        self.recommend = recommend_arr
        self.write_recommend()

        return recommend_arr

    def get_recommend_vectors(self):
        """
        추천 결과를 가져오는 함수
        """
        rec_vector_dict = dict()
        rec_vector_dict['recommend'] = self.recommend
        rec_vector_dict['item_like_prob'] = self.item_like_prob_mat

        return rec_vector_dict

    def write_recommend(self):
        """
        추천 결과 저장 함수
        """
        rec_vector_dict = self.get_recommend_vectors()

        file_path = os.path.dirname(os.path.realpath('__file__'))
        with open(file_path + '/rec_vector_dict.pickle', 'wb') as f:
            pickle.dump(rec_vector_dict, f, pickle.HIGHEST_PROTOCOL)
