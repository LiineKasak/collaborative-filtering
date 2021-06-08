# CIL Project 2021: Collaborative Filtering

## Requirements
### Prepare environment using script
Run `source bin/init.sh` to run locally, or `source bin/leonhard_init.sh` to run on leonhard.
- This will create a virtual environment (if it does not yet exist) and make sure the requirements are satisied. 
- It will also add the working directory `.../collaborative-filtering/src` to PYTHONPATH.


If this doesnt work run manually:
### Prepare environment Manually
1. Add directory to PYTHONPATH: \
    `export PYTHONPATH=/path_to_source_directory/collaborative-filtering/src:$PYTHONPATH`
    
2. create and activate virtual environment \
    `python3 -m venv ./venv` \
    `source venv/bin/activate`
    
3. install requirements \
    `pip install -r requirements.txt`
    
    
For leonhard, execute the first 2 steps, and then run 

`module load python_cpu/3.7.4 eth_proxy`

`pip install -r requirements.txt`

## Proposed Project Structure:
```markdown
bin/
    init.sh
    init_leonhard.sh
data/
     submissions/
     data_train.csv
src/
    experiments/
        notebook.py
        run_xyz.py -- files that are required to actually run the approaches (for reproducible experiments)
    models/ -- one file per method, all implemented according to some interface (see below)
        algobase.py -- every method should inherit from this
            - kaggle_predict
            - predict
            - fit
            - cross_validate
        svd.py
        ncf.py
        ...
    utils/ -- helper functions
        data_processing.py:
            - read the data, and return users, items, ratings lists
            - split into matrix and mask
            - get_statistics
            - get_score
            - create_submission_file


report/ (later)

requirements.txt

.gitignore
README

	
```

**Algorithms-Classes (see `src/algobase.py`):** 

```python
class approach(AlgoBase):
    def __init__(params):
        self.parameters = params # set parameters
    
    def fit(train_data):
        # train the approach
    
    def predict(test_data):
        # predict value for given test-input


	
```
