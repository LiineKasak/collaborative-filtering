from auxiliary import data_processing
from src.autoencoder.deep_autoencoder import DeepAutoEncoder
from src.autoencoder.variational_autoencoder import VAE
from src.autoencoder.collaborative_denoising_autoencoder import CDAE
import numpy as np

data_pd = data_processing.read_data()

dae = DeepAutoEncoder()
vae = VAE()
cdae = CDAE()
auto_encoders = [dae, vae, cdae]

best_index = 0
best_rmse = 100

for i, auto_encoder in enumerate(auto_encoders):
    print(f'Evaluating {auto_encoder.__class__.__name__}...')
    rmses = auto_encoder.cross_validate(data_pd)
    print('RMSES:', rmses)
    mean_rmse = np.mean(np.array(rmses))
    print(f'Mean RMSE {np.mean(np.array(rmses))}')
    if mean_rmse < best_rmse:
        best_index = i
        best_rmse = mean_rmse

best_auto_encoder = auto_encoders[best_index]
users, movies, predictions = data_processing.extract_users_items_predictions(data_pd)
best_auto_encoder.fit(users, movies, predictions)
best_auto_encoder.predict_for_submission(name=best_auto_encoder.__class__.__name)

"""
Output:
Evaluating DeepAutoEncoder...
RMSES: [1.1165483749881933, 1.1137586889569995, 1.1197193232403213, 1.1137113819636275, 1.1153824189949555]
Mean RMSE 1.1158240376288195
Evaluating VAE...
RMSES: [1.1141451340980428, 1.114206267647576, 1.117005508382096, 1.1139286736040113, 1.11243409423196]
Mean RMSE 1.1143439355927374
Evaluating CDAE...
RMSES: [1.1144677893387234, 1.1144963788859443, 1.1172459580975296, 1.1142717263261406, 1.112798436670073]
Mean RMSE 1.1146560578636822
"""
