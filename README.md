# CIL Project 2021: Collaborative Filtering

## Requirements

1. Add directory to PYTHONPATH: \
    `export PYTHONPATH=/path_to_source_directory/collaborative-filtering:$PYTHONPATH`
    
2. create and activate virtual environment \
    `python3 -m venv ./venv` \
    `source venv/bin/activate`
    
3. install requirements \
    `pip install -r requirements.txt`
    
    
For leonhard, execute the first 2 steps, and then run `source init_leonhard.sh`

## Proposed Project Structure:
```markdown
auxiliary/ -- helper functions
    model_selection.py -- cross_validation etc
    
    processing.py:
        - read the data, and return users, items, ratings lists
        - split into matrix and mask
        - get_statistics
        - get_score
        - create_submission_file

data/
     submissions/
     data_train.csv

experiments/
    notebook.py
    run_xyz.py -- files that are required to actually run the approaches (for reproducible experiments)

src/ -- one file per method, all implemented according to some interface (see below)
    svd.py
    ncf.py
    ...


report/ (later)

requirements.txt
init_leonhard.sh

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
    
    def cross_validate(data):
        # not sure, but maybe have this s.t. 
        # we can call cross_validate on an approach directly?

	
```
