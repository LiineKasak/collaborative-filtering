# CIL Project 2021: Collaborative Filtering - Augmented Matrix Factorization

## Requirements and Setup
### Prepare environment using script
Run `source bin/init.sh` to run locally, or `source bin/leonhard_init.sh` to run on leonhard (note: must be run from the project root!).
- This will create a virtual environment (if it does not yet exist) and make sure the requirements are satisied. 
- It will also add the working directory `.../collaborative-filtering/src` to PYTHONPATH.


If this doesnt work run manually:
### Prepare environment Manually
1. Add directory to PYTHONPATH: \
    `export PYTHONPATH=/path_to_source_directory/collaborative-filtering/src:$PYTHONPATH`
    
2. create and activate virtual environment 
   
    `python3 -m venv ./venv` \
    `source venv/bin/activate`
    
3. install requirements \
    `pip install -r requirements.txt`
    
    
For leonhard, execute step 1 and 2, and then 

4. Load the required modules: `module load python_cpu/3.7.4 eth_proxy`

5. Install the requirements manually: `pip install -r requirements.txt`

___
## Data
To be able to run some of the approaches, data has to be downloaded. To do this, visit https://polybox.ethz.ch/index.php/s/IMeZCANjqGhz6kb 
and download the files you'll find there. These are: 
- `phase1_precomputed_matrix.zip`
- `phase2_pretrained_model.zip`

Unpack them and place them inside the `data/` folder.


___
## Reproducing Experiments

To reproduce the results of different models, `main.py` should be run from commandline.

### Models
As a required positional argument, the model must be selected.

| Argument        | Corresponding model                                  | CV score |
|-----------------|------------------------------------------------------|:--------:|
| gmf             | Generalized Matrix Factorization                     | 1.1688   |
| mlp             | Multi-Layer Perceptron                               | 1.1562   |
| ncf             | Neural Collaborative Filtering                       | 1.1474   |
| ae              | Deep Autoencoder                                     | 1.1282   |
| cdae            | Collaborative Denousing Autoencoder                  | 1.0930   |
| vae             | Variational Autoencoder                              | 1.0919   |
| svd             | Plain Singular Value Decomposition (SVD)             | 1.0712   |
| svt             | Singular Value Thresholding (SVT)                    | 1.0389   |
| knn             | K Nearest Neighbours                                 | 1.0290   |
| svd_sgd         | SVD-based MF using Stochastic Gradient Descent (SGD) | 0.9842   |
| log_reg         | Logistic Regression                                  | 0.9831   |
| svt_hybrid      | SVD-based MF using hybrid optimization               | 0.9831   |
| svt_init_hybrid | SVT intialized SVD-based MF with hybrid opt.         | 0.9817   |
| aumf            | Augmented Matrix Factorization (MF)                  | 0.9805   |

### Reproducing Kaggle Submission
```
python src/main.py aumf --mode submit
```

### Commandline help
Example output running ```python main.py --help```:

```
usage: main.py [-h] [--mode {val,cv,submit}] [--train_split TRAIN_SPLIT] [--folds FOLDS] [--submission SUBMISSION]
               [--verbal VERBAL] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--regularization REGULARIZATION]
               [--batch_size BATCH_SIZE] [--device DEVICE] [--k_singular_values K_SINGULAR_VALUES]
               [--enable_bias ENABLE_BIAS] [--n_neighbors N_NEIGHBORS] [--shrink_val SHRINK_VAL]
               {aumf,svd,svd_sgd,log_reg,knn,gmf,mlp,ncf,vae,cdae,ae,svt,svt_hybrid,svt_init_hybrid}

Train (, validate and save) a collaborative filtering model.

positional arguments:
  {aumf,svd,svd_sgd,log_reg,knn,gmf,mlp,ncf,vae,cdae,ae,svt,svt_hybrid,svt_init_hybrid}
                        selected model

optional arguments:
  -h, --help            show this help message and exit
  --mode {val,cv,submit}, -m {val,cv,submit}
                        mode: validate, cross-validate (cv) or train for submission (default: 'val').
  --train_split TRAIN_SPLIT, -split TRAIN_SPLIT
                        Train portion of dataset if mode='val' (default: 0.9). Must satisfy 0 < train_split < 1.
  --folds FOLDS, -f FOLDS
                        Number of folds if mode='cv' (default: 5).
  --submission SUBMISSION
                        Submission file name if mode='submit' (default: 'submission').
  --verbal VERBAL, -v VERBAL
  --epochs EPOCHS, -e EPOCHS
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --regularization REGULARIZATION, -r REGULARIZATION
  --batch_size BATCH_SIZE, -b BATCH_SIZE
  --device DEVICE, -d DEVICE
  --k_singular_values K_SINGULAR_VALUES, -k K_SINGULAR_VALUES
  --enable_bias ENABLE_BIAS, -bias ENABLE_BIAS
  --n_neighbors N_NEIGHBORS, -n N_NEIGHBORS
  --shrink_val SHRINK_VAL, -s SHRINK_VAL
```

### Running mode
The `mode` argument specifies how the model will be trained and validated. By default, `mode='val'`.


| Mode argument | Resulting action                                                                                                                                                                                                      |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `val`         | Model will be trained and validated, where the training dataset is `train_split` (default 0.9) of the whole dataset, and the validation dataset makes up the rest. The ouput is the RMSE score on the validation set. |
| `cv`          | Model will be cross-validated with `folds` number of folds. The RMSE on the validation set for each fold will be reported, as well as the average RMSE.                                                               |
| `submit`      | Model will be trained on the whole dataset and predictions for the kaggle test set will be saved as `${submission}.csv.zip`                                                                                           |

### Equivalence to results in the report
If no other arguments are specified, default fine-tuned hyperparameters are set for each model, and the models will behave exactly as described in our report. However, we leave the option to experiment with models open.

### CF-NADE
The experiments with CF-NADE were run using the original implementation of Zheng et al. which is not compatible with the rest of our code.
The implementation can be found in the CF-NADE directory and should be run with Python 2.
Converting our dataset to the proper MovieLens format used by the CF-NADE implementation can be done with `src/utils/movielens-transform.py`. 
