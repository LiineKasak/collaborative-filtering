# make sure first to add the directory to PYTHONPATH: 
# export PYTHONPATH=/path_to_source_directory/collaborative-filtering:$PYTHONPATH
# python3 -m venv ~/venv

directory_name=`pwd`
export PYTHONPATH=$directory_name/../:$PYTHONPATH

if [ -d "./venv" ]; then
  source venv/bin/activate
else
  python3 -m venv ~/venv
  source venv/bin/activate
fi

module load python_cpu/3.7.4 eth_proxy

source venv/bin/activate

pip install -r requirements.txt
