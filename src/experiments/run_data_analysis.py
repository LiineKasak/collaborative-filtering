from src.utils.dataset import DatasetWrapper
from src.utils.data_analysis import DataAnalyzer
from src.utils import data_processing


data_pd = data_processing.read_data()
data_wrapper = DatasetWrapper(data_pd)

da = DataAnalyzer(data_pd)

# da.create_test_histograms(filename='submissions/svd_sgd_norm_k12_75.csv', with_ratings=True)
da.create_validation_plots('SVD_SGD_validation.csv')