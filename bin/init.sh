project_dir="$PWD"
export PYTHONPATH="$project_dir/src:$PYTHONPATH"

if [ -d "./venv" ]; then
  echo "virtual environment found"
else
  echo "virtual environment not found. Creating ./venv/ now"
  python3 -m venv "$project_dir"/venv
fi

source "$project_dir"/venv/bin/activate

pip install -r requirements.txt