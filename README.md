# mhealth
Mobile health: data analysis and visualization for health data from wearable devices

## Python version
3.8 

## Setup via command line

```bash
git clone https://github.com/hirsch-lab/mhealth.git
cd mhealth
pip install -r requirements.txt
# Run the unit tests to verify installation
pip install pytest
export MHEALTH_DATA="./src/resources"
pytest
```

## Setup in PyCharm

* "File" -> "Open"
* Select git repository mhealth
* Create a new virtual environment (python interpreter): "Configure Python interpreter"
    * "Add Interpreter"
    * "Virtual Environment" -> "New environment" with "venv" directory within your project workspace as location, e.g., mhealth/venv
* Install packages from requirements.txt
    * "Tools" -> "Sync Python Requirements": select requirements.txt and "Strong equality >="
    * Wait until all packages are installed and indexer is updated
* Right-click on "src" folder and "Mark Directory as ... Sources Root"
* Note: virtual environment "venv" folder should be excluded "Mark Directory as ... Excluded"
* Run unit tests

