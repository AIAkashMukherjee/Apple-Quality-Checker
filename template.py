import os
from pathlib import Path
import logging

list_of_files=[
    '.github/.workflows/.gitkeep', # for pushing file to github
    'src/__init__.py',
    'src/components/__init__.py',
    'src/components/data_ingestion.py',
    'src/components/data_transformation.py',
    'src/components/model_trainer.py',
    'src/components/model_evalutaion.py',
    'src/pipeline/__init__.py',
    'src/pipeline/training_pipeline.py',
    'src/pipeline/prediction_pipeline.py',
    'src/utlis/__init__.py',
    'src/utlis/utlis.py',
    'src/logger/custom_logging.py',
    'src/logger/__init__.py',
    'src/exceptions/expection.py',
    'src/exceptions/__init__.py',
    'src/entity/__init__.py',
    'src/entity/config_entity.py'
    'setup.py',
    'setup.config',

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating Directory {filedir} for file {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file