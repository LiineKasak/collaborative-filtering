import argparse

import os
from sklearn.model_selection import KFold
from tqdm import tqdm

from models.algobase import AlgoBase
from utils.dataset import DatasetWrapper
from utils import data_processing
from models.svt_init_svd_als_sgd_hybrid import SVT_INIT_SVD_ALS_SGD
from models.gmf import GMF
import pickle


class AuMF(AlgoBase):
    """ Augmented Matrix Factorization """

    def __init__(self, params: argparse.Namespace, use_pretrained_svd=True):
        AlgoBase.__init__(self)
        svt_params = SVT_INIT_SVD_ALS_SGD.default_params()
        self.svt_hybrid = SVT_INIT_SVD_ALS_SGD(svt_params)
        self.gmf = None

        self.use_pretrained_svd = use_pretrained_svd

        self.device = params.device
        self.epochs = params.epochs

        self.batch_size = params.batch_size
        self.lr = params.learning_rate

        self.fold = '_no_cv'
        self.svt_precompute_path = None

    @staticmethod
    def default_params():
        return argparse.Namespace(epochs=5, batch_size=256, learning_rate=0.01, device="cpu",
                                  verbal=True)

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        # path of the pretrained model (if it exists)
        self.svt_precompute_path = data_processing.get_project_directory() + f'{os.sep}data{os.sep}phase2_pretrained_model{os.sep}svt_precomputed{os.sep}svt_advanced_{self.svt_hybrid.k}{self.fold}.pickle'
        if not self.use_pretrained_svd:
            self.svt_hybrid.fit(train_data)
            self.svt_hybrid.save(self.svt_precompute_path)  # export model

        svd = pickle.load(open(self.svt_precompute_path, 'rb'))

        self.gmf = GMF(user_embedding=svd.pu,
                       movie_embedding=svd.qi,
                       user_bias=svd.bu,
                       movie_bias=svd.bi,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       learning_rate=self.lr,
                       device=self.device)

        self.gmf.fit(train_data=train_data, test_data=test_data)

    def predict(self, users, movies):
        predictions = self.gmf.predict(users, movies)

        return predictions

    def cross_validate(self, data_pd, folds=5, random_state=42):
        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)

        rmses = []

        bar = tqdm(total=folds, desc='cross_validation')
        counter = 0
        k = self.svt_hybrid.k

        for train_index, test_index in kfold.split(data_pd):
            self.fold = f"_fold{counter}"

            # load train and validation data
            train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[train_index])
            val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[test_index])
            train_data = DatasetWrapper(train_users, train_movies, train_predictions)
            val_data = DatasetWrapper(val_users, val_movies, val_predictions)

            # load initialization matrix for the svt-model
            self.svt_hybrid.svt_init_matrix_path = '/data/phase1_precomputed_matrix/' + \
                                                   self.svt_hybrid.cv_svt_matrix_filenames[counter]

            # fit
            self.fit(train_data=train_data, test_data=val_data)

            # predict and compute scores
            predictions = self.predict(val_users, val_movies)

            data_processing.create_validation_file(users=val_users, movies=val_movies,
                                                   ground_truth_predicitons=val_predictions, predictions=predictions,
                                                   name=f'aumf_predictions_fold{counter}')
            rmse = data_processing.get_score(predictions, val_predictions)

            rmses.append(rmse)

            # update counters
            counter += 1
            bar.update()

        bar.close()

        return rmses
