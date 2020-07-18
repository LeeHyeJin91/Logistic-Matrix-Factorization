import os
import pickle

class Utils():

    def __init__(self):
        pass

    def read_vectors(self):
        """
        저장된 모델 파라미터를 불러오는 함수
        """
        file_path = os.path.dirname(os.path.realpath('__file__'))
        with open(file_path + '/vector_dict.pickle', 'rb') as f:
                vector_dict = pickle.load(f)

        return vector_dict

    def read_recommend(self):
        """
        저장된 추천 결과를 불러오는 함수
        """
        file_path = os.path.dirname(os.path.realpath('__file__'))
        with open(file_path + '/rec_vector_dict.pickle', 'rb') as f:
                rec_vector_dict = pickle.load(f)

        return rec_vector_dict
