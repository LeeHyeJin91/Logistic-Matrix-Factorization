import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.utils.validation import check_random_state

class Split():

    def make_csr_matrix(self, data, rows, cols, dtype=np.float64):
        """
        csr matrix를 만드는 함수

        rows: row index list
        cols: col index list
        data: value list
        """
        data = np.array(data)
        rows = np.array(rows)
        cols = np.array(cols)

        shape = (len(np.unique(rows)), len(np.unique(cols)))
        csr_matrix =  sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=dtype)

        return csr_matrix

    def make_train_mask(self, users, items, train_mask_threshold, random_state):
        """
        train mask를 만드는 함수

        users: user index list
        items: item index list
        train_mask_threshold: 0과 1사이의 실수
        random_state: seed 값
        """
        users = np.array(users)
        items = np.array(items)

        random_state = check_random_state(random_state)
        n_events = users.shape[0]
        train_mask = random_state.rand(n_events) <= train_mask_threshold #[True, False, ...]

        # goal: test데이터가 train에 모두 있어야함
        for array in (users, items):

            train_vals = array[train_mask]
            test_vals = array[~train_mask]

            missing_vals = np.unique(test_vals[np.where(~np.in1d(test_vals, train_vals))[0]]) #train에 없는 test값 ex.[12,45]
            if missing_vals.shape[0] == 0:
                continue
            missing_vals_check = np.in1d(array, missing_vals)  #array에서 [12,45]에 해당하는 위치 ex. [1,0,1,0,0,1]
            missing_vals_idx = np.where(missing_vals_check)[0] #array에서 [12,45]에 해당하는 인덱스 ex. [0,2,5]

            added = set()
            for idx in missing_vals_idx:
                val = array[idx]
                if val in added:
                    continue
                train_mask[idx] = True
                added.add(val)

        return train_mask

    def make_train_test(self, users, items, ratings, train_mask):
        """
        train, test 데이터를 만들어주는 함수

        users: user index list
        items: item index list
        ratings: rating value list
        train_mask: train mask list ex.[1,0,0,1]
        """
        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings)

        # train
        train = self.make_csr_matrix(data=ratings[train_mask], rows=users[train_mask], cols=items[train_mask], dtype=np.float64)
        # test
        test = np.zeros((len(np.unique(users)), len(np.unique(items))))
        for te_u, te_i, te_r in zip(users[~train_mask], items[~train_mask], ratings[~train_mask]):
            test[te_u, te_i] = te_r
        test = sparse.csr_matrix(test)

        return train, test

    def train_test_split(self, users, items, ratings, train_mask_threshold, random_state):
        """
        train, test를 나눠주는 함수
        """
        train_mask = self.make_train_mask(users, items, train_mask_threshold, random_state)
        train, test = self.make_train_test(users, items, ratings, train_mask)
        print('train_size:', round(sum(train_mask)/len(users), 1))

        return train, test
