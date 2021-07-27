import numpy as np
from sklearn.model_selection import train_test_split
from utils import data_processing
from models.algobase import AlgoBase
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from scipy import sparse

EPSILON = 1e-5


class SVT(AlgoBase):
    """
    Running Singular Value Thresholding Algorithm.
    By "A Singular Value Thresholding Algorithm for Matrix Completion":
    https://arxiv.org/pdf/0810.3286.pdf
    """

    def __init__(self, k_singular_values=12, shrink_val = 30000, max_it=150, learning_rate=1.99, verbal=False,
                 track_to_comet=False, useInitMatrix=False):
        AlgoBase.__init__(self, track_to_comet)

        self.max_it = max_it
        self.shrink_val = shrink_val
        self.learning_rate = learning_rate
        self.verbal = verbal
        self.k_singular_values = k_singular_values
        self.useInitMatrix = useInitMatrix

        self.matrix = np.zeros((self.number_of_users, self.number_of_movies))
        self.reconstructed_matrix = np.zeros((self.number_of_users, self.number_of_movies))

        self.Xopt = np.zeros((self.number_of_users, self.number_of_movies))
        self.Yopt = np.zeros((self.number_of_users, self.number_of_movies))
        self.Y0 = np.zeros((self.number_of_users, self.number_of_movies))


    def _update_reconstructed_matrix(self):
        U, s, Vt = np.linalg.svd(self.Xopt, full_matrices=False)

        S = np.zeros((self.number_of_movies, self.number_of_movies))
        S[:self.k_singular_values, :self.k_singular_values] = np.diag(s[:self.k_singular_values])
 
        self.reconstructed_matrix = U.dot(S).dot(Vt)
        print(self.reconstructed_matrix[0][0])

    def fit(self, users, movies, ground_truth, valid_users=None, valid_movies=None, valid_ground_truth=None):
        self.data, self.mask = data_processing.get_data_mask(users, movies, ground_truth)

        run_validation = valid_users is not None and valid_movies is not None and valid_ground_truth is not None

        time_string = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f'./logs/SVT_{time_string}'
        writer = SummaryWriter(log_dir)

        if (self.useInitMatrix):
            with open('svt_Xopt_Yk.npy', 'rb') as f:
                self.Xk = np.load(f, allow_pickle=True)
                self.Yk = np.load(f, allow_pickle=True)
        else:
            self.Yk = self.Y0

        with tqdm(total=self.max_it * len(users), disable=not self.verbal) as pbar:
            for iter in range(self.max_it):
                self.Yk = sparse.csr_matrix(self.Yk, dtype=float)
                U, s, Vt = sparse.linalg.svds(self.Yk, k=999)

                self.Xk = np.zeros((self.number_of_users, self.number_of_movies))

                for i in range(998,-1,-1):
                    if(s[i] > self.shrink_val):
                        self.Xk += (s[i]-self.shrink_val)*(np.outer(U[:,i], Vt[i,:]))
                    else:
                        break
                diff = np.linalg.norm(self.mask*(self.Xk-self.data))/np.linalg.norm(self.data)
                if(diff <= EPSILON):
                    break
                self.Yk = self.Yk + self.learning_rate*(self.mask*((self.data-self.Xk)))
                self.Xopt = self.Xk
                self.Yopt = self.Yk
            
                if (iter % 10 == 0):
                    self._update_reconstructed_matrix()
                    predictions = self.predict(users, movies)
                    rmse_loss = data_processing.get_score(predictions, ground_truth)
                    writer.add_scalar('rmse', rmse_loss, iter)
                    valid_predictions = self.predict(valid_users, valid_movies)
                    reconstruction_rmse = data_processing.get_score(valid_predictions, valid_ground_truth)
                    writer.add_scalar('val_rmse', reconstruction_rmse, iter)

            self._update_reconstructed_matrix()
            with open('svt_Xopt_Yk.npy', 'wb') as f:
                np.save(f, self.Xopt, allow_pickle=True)
                np.save(f, self.Yopt, allow_pickle=True)

    def predict(self, users, movies):
        predictions = data_processing.extract_prediction_from_full_matrix(self.reconstructed_matrix, users, movies)
        predictions[predictions > 5] = 5
        predictions[predictions < 1] = 1
        return predictions
    
    def cross_validate(self, data_pd, folds=5, random_state=42):
        """ Run Crossvalidation using kfold, taking a pandas-dataframe of the raw data as input
            (as it is read in from the .csv file) """
        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)

        rmses = []
        counter = 0

        cv_svt_matrix_filenames = ['svt_Xopt_Yk_sh100k_0_to_2000_CV_20210723-005714.npy',
        'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210723-080627.npy',
        'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210723-151440.npy',
        'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210723-222641.npy',
        'svt_Xopt_Yk_sh100k_0_to_2000_CV_20210724-054423.npy']

        bar = tqdm(total=folds,  desc='cross_validation')
        
        for train_index, test_index in kfold.split(data_pd):
            train_users, train_movies, train_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[train_index])
            val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(
                data_pd.iloc[test_index])

            with open('/data/phase1_precomputed_matrix/' + cv_svt_matrix_filenames[counter], 'rb') as f:
                self.Xopt =  np.load(f, allow_pickle=True)
                self.Y0 = np.load(f, allow_pickle=True)

            self._update_reconstructed_matrix()
            counter += 1
            
            predictions = self.predict(val_users, val_movies)
            rmses.append(data_processing.get_score(predictions, val_predictions))

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
    shrink_val = 100000
    max_it= 500
    k_singular_values = 12
    svt = SVT(k_singular_values=k_singular_values, shrink_val=shrink_val, max_it=max_it)

    submit = False

    if submit:
        users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
        svt.fit(users, movies, predictions)
        svt.predict_for_submission(f'svt{shrink_val}_{max_it}')
    else:
        rmses = svt.cross_validate(data_pd)
        print("RMSES of ", svt.method_name, "\n", rmses, "\n")
