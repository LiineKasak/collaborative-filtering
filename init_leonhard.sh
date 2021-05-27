# make sure first to add the directory to PYTHONPATH: 
# export PYTHONPATH=/path_to_source_directory/collaborative-filtering:$PYTHONPATH
# python3 -m venv ~/venv

module load python_cpu/3.7.4 eth_proxy

source venv/bin/activate

pip install -r requirements.txt
