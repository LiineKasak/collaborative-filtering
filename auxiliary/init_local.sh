directory_name=`pwd`
export PYTHONPATH=$directory_name/../:$PYTHONPATH

if [ -d "./venv" ]; then
  source venv/bin/activate
else
  python3 -m venv ~/venv
  source venv/bin/activate
fi 

pip install -r requirements.txt





