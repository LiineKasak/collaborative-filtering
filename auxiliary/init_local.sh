directory_name=`pwd`
export PYTHONPATH=$directory_name/../:$PYTHONPATH

if [ -d "./venv" ]; then
  echo "virtual environment found"
else
  echo "virtual environment not found. Creating ./venv/ now"
  python3 -m venv ~/venv
fi

source venv/bin/activate

pip install -r requirements.txt





