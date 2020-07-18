import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

class Metric():

    def auc(self, user_item_mat, item_like_prob_mat):
        """
        auc 계산 함수
        """
        sample_idx = np.random.choice(user_item_mat.shape[0], item_like_prob_mat.shape[0], replace=False)

        auc = 0.0
        num_users, num_items = user_item_mat.shape
        for user_idx, user_data in enumerate(user_item_mat):
            try:
                y_pred = item_like_prob_mat[user_idx]
                y_true =  np.array([1 if ud!=0 else 0 for ud in user_data])
                u_auc = roc_auc_score(y_true, y_pred)
                auc += u_auc
            except:
                pass

        auc /= num_users

        return auc

    def calculate_percentile_ranking(self, item_like_prob_mat):
        """
        percentile_ranking 계산함수
        """
        df = pd.DataFrame({'prob': item_like_prob_mat})
        df['perc_rank'] = df['prob'].rank(pct=True)
        perc_ranking_lst = list(df['perc_rank'])

        return perc_ranking_lst

    def mean_percentile_ranking(self, user_item_mat, item_like_prob_mat):
        """
        mean_percentile_ranking 계산함수
        """
        mpr = 0.0
        num_users, num_items = user_item_mat.shape
        for user_idx, (user_data, prob_lst) in enumerate(zip(user_item_mat, item_like_prob_mat)):
            y_true = user_data
            y_pred = self.calculate_percentile_ranking(prob_lst)

            u_mpr = np.sum(np.array(y_pred)*np.array(y_true))/np.sum(y_true)
            if np.isnan(u_mpr) == False:
                mpr += u_mpr
        mpr /= num_users

        return mpr

