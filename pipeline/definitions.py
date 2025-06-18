import os
from pathlib import Path

pathroot = os.path.abspath(os.path.dirname(__file__))

PROJECT = str(Path(__file__).resolve().parents[1]).replace('\\','/')

# Setup data directory if not exist
Path(os.path.join(PROJECT, 'data')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/datasets')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/datasets/real_data')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/datasets/sim_data')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/datasets/raw_data')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/datasets/sim_data/from_generator')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/weather')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/weather/reference_weathers')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/anagrafiche')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/datasets/filtered_data')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/datasets/clean_data')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'data/datasets/normalized_data')).mkdir(parents=True, exist_ok=True)

Path(os.path.join(PROJECT, 'models')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'models/parametric')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'models/semiParametric')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'models/blackbox')).mkdir(parents=True, exist_ok=True)

Path(os.path.join(PROJECT, 'results')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(PROJECT, 'figures')).mkdir(parents=True, exist_ok=True)

DATA = os.path.join(PROJECT, 'data').replace('\\','/')
DATASETS=os.path.join(DATA, 'datasets').replace('\\','/')
ANAGRAFICHE = os.path.join(DATA, 'anagrafiche').replace('\\','/')
REF_WEATHERS = os.path.join(DATA, 'weather/reference_weathers').replace('\\','/')
REAL_DATA=os.path.join(DATASETS, 'real_data').replace('\\', '/')
SIM_DATA=os.path.join(DATASETS, 'sim_data').replace('\\', '/')
SIM_DATA_NO_FORMAT=os.path.join(DATASETS, 'sim_data/from_generator').replace('\\', '/')
DATA_RAW=os.path.join(DATASETS, 'raw_data').replace('\\', '/')
DATA_CLEAN=os.path.join(DATASETS, 'clean_data').replace('\\', '/')
DATA_FILTERED=os.path.join(DATASETS, 'filtered_data').replace('\\', '/')
DATA_NORMALIZED=os.path.join(DATASETS, 'normalized_data').replace('\\', '/')

MODELS=os.path.join(PROJECT, 'models').replace('\\', '/')
PAR_MODELS=os.path.join(MODELS, 'parametric').replace('\\', '/')
SEMIPAR_MODELS=os.path.join(MODELS, 'semiParametric').replace('\\', '/')
BLACKBOX_MODELS=os.path.join(MODELS, 'blackbox').replace('\\', '/')

RESULTS=os.path.join(PROJECT, 'results').replace('\\', '/')
FIGURES=os.path.join(PROJECT, 'figures').replace('\\', '/')

PLOT = os.path.join(PROJECT, 'plot').replace('\\','/')