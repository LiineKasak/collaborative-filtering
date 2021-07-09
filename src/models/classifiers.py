import sklearn
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from models.adapted_svd_sgd import ADAPTED_SVD_SGD
from utils import data_processing, dataset, data_analysis
from models.algobase import AlgoBase
import numpy as np
import pandas as pd

from models.svd import SVD
from models.svd_sgd import SVD_SGD


class svdInputClassifier(AlgoBase):
    def __init__(self, k=12, epochs=75, clf=None, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        """ - initialize the method (number of users/movies, and the method name).
            - initialize the comet experiment if desired (default is no tracking)
            - if you want to track to a different comet workspace, you can pass arguments to it."""

        AlgoBase.__init__(self, track_to_comet)
        self.datawrapper = None
        self.user_embeddings, self.movie_embeddings = None, None

        if clf is None:
            self.clf = MLPClassifier(hidden_layer_sizes=(100, 100),
                                     activation='relu',
                                     solver='adam',
                                     alpha=0.0001,
                                     batch_size='auto',
                                     # learning_rate='constant',    # only used with sgd
                                     learning_rate_init=0.001,
                                     # power_t=0.5,                  # only used if learning_rate='invscaling'
                                     max_iter=200,
                                     shuffle=True,
                                     random_state=42,
                                     tol=0.0001,
                                     verbose=True,
                                     warm_start=False,
                                     momentum=0.9,
                                     # nesterovs_momentum=True,  # only with sgd
                                     early_stopping=True,
                                     validation_fraction=0.1,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-08,
                                     n_iter_no_change=10,
                                     max_fun=15000)

            # self.clf = AdaBoostClassifier() # sucks

        else:
            self.clf = clf


        self.sgd = ADAPTED_SVD_SGD(verbal=True, epochs=epochs, k_singular_values=k, use_prestored=True, store=False)

    def predict(self, users, movies):
        X = self.get_features(users, movies)
        probs = self.clf.predict_proba(X)
        ratings = np.array([1,2,3,4,5])

        pred = probs * ratings

        return np.sum(pred, axis=1)

    def fit(self, datawrapper, val_users=None, val_movies=None):
        self.datawrapper = datawrapper
        users, movies, ratings = datawrapper.users, datawrapper.movies, datawrapper.ratings
        print("...fitting the sgd")

        self.sgd.fit(datawrapper, val_users, val_movies)
        # self.user_embeddings, self.movie_embeddings = SVD.get_embeddings(12, self.datawrapper.data_matrix)

        X = self.get_features(users, movies)

        y = ratings

        # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        # sel.fit_transform(X)
        # print(sel._get_support_mask())
        print("...fitting the classifier")
        self.clf.fit(X, y)



    def get_features(self, users, movies):
        user_features = self.sgd.pu[users]
        movie_features = self.sgd.qi[movies]
        user_means = np.reshape(self.datawrapper.user_means[users], (-1, 1))
        movie_means = np.reshape(self.datawrapper.movie_means[movies], (-1, 1))
        user_bias = np.reshape(self.sgd.bu[users], (-1, 1))
        movie_bias = np.reshape(self.sgd.bi[movies], (-1, 1))

        user_var = np.reshape(self.datawrapper.user_variance[users], (-1, 1))
        movie_var = np.reshape(self.datawrapper.movie_variance[movies], (-1, 1))

        num_movies_watched = np.reshape(self.datawrapper.num_movies_watched[users], (-1, 1))
        times_watched = np.reshape(self.datawrapper.times_watched[movies], (-1, 1))

        X = np.concatenate((user_features, movie_features, user_means, movie_means, user_bias, movie_bias, user_var, movie_var, num_movies_watched, times_watched), axis=1)
        return X


class BaysianClassifier(AlgoBase):

    def __init__(self, track_to_comet=False, method_name=None, api_key="rISpuwcLQoWU6qan4jRCAPy5s",
                 projectname="cil-experiments", workspace="veroniquek", tag="baseline"):
        """ - initialize the method (number of users/movies, and the method name).
            - initialize the comet experiment if desired (default is no tracking)
            - if you want to track to a different comet workspace, you can pass arguments to it."""

        AlgoBase.__init__(self, track_to_comet)
        self.data_wrapper = None

        # self.clf = MultinomialNB(class_prior=np.array([0.0370037 , 0.08430843, 0.2330233 , 0.27582758, 0.36983698])) #TODO: add class_prior
        self.clf = MultinomialNB(fit_prior=False)
    def predict(self, users, movies):
        user_means = self.data_wrapper.user_means[users]
        movie_means = self.data_wrapper.movie_means[movies]
        X = np.array(list(zip(users, movies, user_means, movie_means)))
        pred = self.clf.predict(X)
        return pred

    def fit(self, datawrapper, val_users=None, val_movies=None):
        self.data_wrapper = datawrapper

        users, movies, ratings = datawrapper.users, datawrapper.movies, datawrapper.ratings
        user_means = datawrapper.user_means[users]
        movie_means = datawrapper.movie_means[movies]
        X = np.array(list(zip(users, movies, user_means, movie_means)))
        y = ratings

        self.clf.fit(X, y)

    if __name__ == '__main__':
        use_prestored = True
        if use_prestored:
            k = 12
            epochs = 75

            model = svdInputClassifier(epochs=epochs, k=k)

            # k = 100
            directory_path = data_processing.get_project_directory()
            storing_directory = f"{directory_path}/data/precomputed_svd_e{epochs}_k{k}"

            train_pd = pd.read_csv(f"{storing_directory}/train_pd.csv")
            test_pd = pd.read_csv(f"{storing_directory}/test_pd.csv")

            train_dw = dataset.DatasetWrapper(train_pd)
            test_dw = dataset.DatasetWrapper(test_pd)

            test_users, test_movies, test_predictions = test_dw.users, test_dw.movies, test_dw.ratings

            model.fit(train_dw)
            pred = model.predict(test_users, test_movies)
            rmse = data_processing.get_score(pred, test_predictions)
            print(f"score: {rmse}")
            pa = data_analysis.PredictionAnalyser(test_movies=test_movies, test_predictions=test_predictions,
                                                  test_users=test_users, output_predictions=pred)

            pa.analyze_prediction()

            data_processing.create_validation_file(test_users, test_movies, pred, test_predictions)
            da = data_analysis.DataAnalyzer(train_pd)
            da.create_validation_histograms(pred, test_predictions)
            da.create_validation_scatterplot(pred, test_predictions)

        else:
            data_pd = data_processing.read_data()

            model = svdInputClassifier()

            train_pd, test_pd = train_test_split(data_pd, train_size=0.9, random_state=42)
            train_pd, val_pd = train_test_split(train_pd, train_size=0.9, random_state=42)
            data_wrapper = dataset.DatasetWrapper(train_pd)
            # users, movies, predictions = data_processing.extract_users_items_predictions(train_pd)

            # rsmes = sgd.cross_validate(data_pd)
            # print("Cross validations core: ", np.mean(rsmes))
            val_users, val_movies, val_predictions = data_processing.extract_users_items_predictions(val_pd)
            test_users, test_movies, test_predictions = data_processing.extract_users_items_predictions(test_pd)
            model.fit(data_wrapper)

            pred = model.predict(test_users, test_movies)

            rmse = data_processing.get_score(pred, test_predictions)
            print(f"score: {np.mean(rmse)}")

            # pa = data_analysis.PredictionAnalyser(test_movies=test_movies, test_predictions=test_predictions,
            #                                       test_users=test_users, output_predictions=pred)
            #
            # pa.analyze_prediction()
            #
            # data_processing.create_validation_file(val_users, val_movies, pred, val_predictions)
            # da = data_analysis.DataAnalyzer(data_pd)
            # da.create_validation_histograms(pred, val_predictions)
            # da.create_validation_scatterplot(pred, val_predictions)



