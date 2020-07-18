import os
import pandas as pd
import numpy as np
from Split import Split
from LogisticMF import LogisticMF
from Metric import Metric
from Recommend import Recommend

file_path = os.path.dirname(os.path.realpath('__file__'))

class Run:

    def __init__(self):

        # 데이터 불러오기
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(file_path + '/u.data', sep = '\t', names = names)
        #df = pd.read_csv('C:/Users/user00/Desktop/ml-100k/u.data', sep = '\t', names = names)
        df['user_id'] = [uid-1 for uid in df['user_id']]
        df['item_id'] = [iid-1 for iid in df['item_id']]

        # user-item-count array 생성
        num_users = len(np.unique(df['user_id']))
        num_items = len(np.unique(df['item_id']))
        num_factors = 15

        self.user_item_cnt_lst = []
        for user, item, cnt in zip(df['user_id'], df['item_id'], df['rating']):
            usc = [user, item, cnt]
            self.user_item_cnt_lst.append(usc)

    def run(self):

        # train,test split
        users = np.array([uic[0] for uic in self.user_item_cnt_lst])
        items = np.array([uic[1] for uic in self.user_item_cnt_lst])
        ratings = np.array([uic[2] for uic in self.user_item_cnt_lst])

        split = Split()
        train, test = split.train_test_split(users, items, ratings, 0.8, 1234)

        # train logistic mf
        logmf = LogisticMF(train.toarray(), 15, reg_param=0.6, gamma=1.0, iterations=30)
        logmf.apply_alpha
        logmf.train_model()
        vector_dict = logmf.get_vectors()

        # recommend
        rec = Recommend(train.toarray(), vector_dict)
        recommend = rec.recommend(10) # 추천 결과
        rec_vector_dict = rec.get_recommend_vectors()

        # metric
        tr = train.toarray()
        te = test.toarray()
        item_like_prob_mat = rec_vector_dict['item_like_prob']

        metric = Metric()
        size =  tr.shape[0]

        sample_idx = np.random.choice(tr.shape[0], size, replace=False)
        mpr_tr = metric.mean_percentile_ranking(tr[sample_idx], item_like_prob_mat[sample_idx])
        mpr_te = metric.mean_percentile_ranking(te[sample_idx], item_like_prob_mat[sample_idx])
        auc_tr = metric.auc(tr, item_like_prob_mat)
        auc_te = metric.auc(te[sample_idx], item_like_prob_mat[sample_idx])
        metric_dict = {'mpr_train':mpr_tr, 'mpr_test':mpr_te, 'auc_train':auc_tr, 'auc_test':auc_te}

        return vector_dict, rec_vector_dict, metric_dict

if __name__ == '__main__':

    logistic_mf = Run()
    vector_dict, rec_vector_dict, metric_dict = logistic_mf.run()

    print(metric_dict)



