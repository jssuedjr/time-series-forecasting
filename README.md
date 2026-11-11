Create a virtual environment and install the project from the repository root.

Follow these steps if running for the first time:
In terminal, run from repository root: 

"
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python -m pip install -e ".[dev]"

python -m ipykernel install --user \
    --name ams586 \
    --display-name "Python (AMS 586)"
"