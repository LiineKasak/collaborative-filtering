#@title Download data

import json
""""
download kaggle dataset. After running init_kaggle(), run
!mv kaggle.json ~/.kaggle/kaggle.json

!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c cil-collaborative-filtering-2021

!unzip data_train.csv.zip 
"""

def init_kaggle():
    kaggle_username = "vroniquekaufmann" #@param {type:"string"}
    kaggle_api_key = "d2de3b36e1a600033e233a0cbd4e7fdf" #@param {type:"string"}

    assert len(kaggle_username) > 0 and len(kaggle_api_key) > 0

    api_token = {"username": kaggle_username,"key": kaggle_api_key}

    with open('kaggle.json', 'w') as file:
        json.dump(api_token, file)


