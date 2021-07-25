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
- phase1_precomputed_matrix.zip
- phase2_pretrained_model.zip

Unpack them and place them inside the `data/` folder.


___
## Reproduce Experiments
### Crossvalidation scores
To run the crossvalidation scores for all baseline approaches, run `python src/experiments/run_crossvalidation.py`. 
This will run the cross-validation and report the average rmse 
for each of the appraoches listed in the paper.