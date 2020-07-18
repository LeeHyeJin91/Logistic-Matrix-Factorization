import os
import time
import pickle
import numpy as np

class LogisticMF():

    def __init__(self, user_item_mat, num_factors, reg_param=0.6, gamma=1.0, iterations=30):

        self.user_item_mat = user_item_mat
        self.num_users = user_item_mat.shape[0]
        self.num_items = user_item_mat.shape[1]

        self.num_factors = num_factors
        self.iterations = iterations

        self.reg_param = reg_param
        self.gamma = gamma

    def apply_alpha(self):
        """
        튜닝파라미터 alpha를 계산한 후, 데이터셋에 반영하는 함수
        """
        total = 0
        num_zeros = 0
        for i in range(self.num_users):
            total += len(self.user_item_mat[i].indices)
            num_zeros += self.num_items - len(self.user_item_mat[i].indices)
        alpha = num_zeros / total
        self.user_item_mat *= alpha

    def train_model(self):
        """
        모델 train 함수
        """
        self.ones = np.ones((self.num_users, self.num_items))

        self.user_vectors = np.random.normal(size=(self.num_users, self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items, self.num_factors))

        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        user_vec_deriv_sum = np.zeros((self.num_users, self.num_factors))
        item_vec_deriv_sum = np.zeros((self.num_items, self.num_factors))

        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))

        for i in range(self.iterations):
            t0 = time.time()

            # Fix items
            user_vec_deriv, user_bias_deriv = self.deriv(True)
            user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)

            vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)

            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv

            # Fix users
            item_vec_deriv, item_bias_deriv = self.deriv(False)
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)

            vec_step_size = self.gamma / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(item_bias_deriv_sum)

            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv
            t1 = time.time()

            print ('iteration %i finished in %f seconds' % (i + 1, t1 - t0))

        self.write_vectors()

    def deriv(self, user):
        """
        objective func를 각 파라미터에 대해 미분하는 함수
        """
        if user:
            vec_deriv = np.dot(self.user_item_mat, self.item_vectors)
            bias_deriv = np.expand_dims(np.sum(self.user_item_mat, axis=1), 1)
        else:
            vec_deriv = np.dot(self.user_item_mat.T, self.user_vectors)
            bias_deriv = np.expand_dims(np.sum(self.user_item_mat, axis=0), 1)

        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        A = np.exp(A)
        A /= (A + self.ones)
        A = (self.user_item_mat + self.ones) * A

        if user:
            vec_deriv -= np.dot(A, self.item_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)
            vec_deriv -= self.reg_param * self.user_vectors
        else:
            vec_deriv -= np.dot(A.T, self.user_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
            vec_deriv -= self.reg_param * self.item_vectors

        return (vec_deriv, bias_deriv)

    def get_log_likelihood(self):
        """
        사후확률을 구하는 함수
        """
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.user_item_mat
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        A = (self.user_item_mat + self.ones) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.item_vectors))

        return loglik

    def get_vectors(self):
        """
        학습 결과로부터, 모델의 파라미터를 가져오는 함수
        """
        vector_dict = dict()
        vector_dict['user_vec'] = self.user_vectors #user latent vector
        vector_dict['item_vec'] = self.item_vectors #item latent vector
        vector_dict['user_bias'] = self.user_biases
        vector_dict['item_bias'] = self.item_biases
        vector_dict['info'] = {'user_num':self.num_users, 'item_num':self.num_items}

        return vector_dict

    def write_vectors(self):
        """
        파라미터 저장함수
        """
        vector_dict = self.get_vectors()

        file_path = os.path.dirname(os.path.realpath('__file__'))
        with open(file_path + '/vector_dict.pickle', 'wb') as f:
            pickle.dump(vector_dict, f, pickle.HIGHEST_PROTOCOL)
