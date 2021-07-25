import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.models.algobase import AlgoBase
from src.utils.dataset import DatasetWrapper
from src.utils import data_processing
from src.models.svt_init_svd_als_sgd_hybrid import SVT_INIT_SVD_ALS_SGD
from src.models.gmf import GMF
from utils.experiment import run_experiment
import pickle

class AuMF(AlgoBase):
    """ Augmented Matrix Factorization """
    def __init__(self, epochs=10, batch_size=256, learning_rate=0.01, device="cpu", track_to_comet=False):
        AlgoBase.__init__(self, track_to_comet)

        self.svt_hybrid = SVT_INIT_SVD_ALS_SGD(verbal=True)
        self.gmf = None


        self.use_pretrained_svd = True

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate

        self.fold = '_no_cv'
        self.svt_precompute_path = None


    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        # path of the pretrained model (if it exists)
        self.svt_precompute_path = data_processing.get_project_directory() + f'/data/phase2_pretrained_model/svt_advanced_{self.svt_hybrid.k}{self.fold}.pickle'

        if not self.use_pretrained_svd:
            self.svt_hybrid.fit(train_data)
            self.svt_hybrid.save(self.svt_precompute_path)  # export model


        svd = pickle.load(open(self.svt_precompute_path, 'rb'))
        self.gmf = GMF(
                svd.pu,
                svd.qi,
                svd.bu,
                svd.bi,
                num_epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.lr,
                device=self.device,
            )

        self.gmf.fit(train_data=train_data, test_data=test_data)

    def predict(self, users, movies):
        predictions = self.gmf.predict(users, movies)

        return predictions

    def cross_validate(self, data_pd, folds=5, random_state=42):
        """ Run Crossvalidation using kfold, taking a pandas-dataframe of the raw data as input
                  (as it is read in from the .csv file)

        """
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
            self.svt_hybrid.svt_init_matrix_path = '/data/phase1_precomputed_matrix/' + self.svt_hybrid.cv_svt_matrix_filenames[counter]

            # fit
            self.fit(train_data=train_data, test_data=val_data)

            # predict and compute scores
            predictions = self.predict(val_users, val_movies)
            rmse = data_processing.get_score(predictions, val_predictions)

            rmses.append(data_processing.get_score(predictions, val_predictions))
            print(f"rsme: {rmse}")

            # update counters
            counter += 1
            bar.update()

        bar.close()

        mean_rmse = np.mean(rmses)
        # track mean rmses to comet if we are tracking
        if self.track_on_comet:
            self.comet_experiment.log_metrics(
                {
                    "root_mean_squared_error": mean_rmse
                }
            )
        print(rmses)
        return rmses

if __name__ == '__main__':
    data_pd = data_processing.read_data()
    k = 12
    epochs = 43

    submit = False

    aumf = AuMF()

    if submit:
        data = DatasetWrapper(data_pd)
        aumf.fit(data)

        aumf.predict_for_submission('aumf')
    else:
        rmses = aumf.cross_validate(data_pd)
        print("RMSES of ", aumf.method_name, "\n", rmses, "\n")