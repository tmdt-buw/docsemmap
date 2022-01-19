# DocSemMap

Welcome to DocSemMap, a pipeline for semantic labeling using textual data descriptions.

## Installation

Create a Python 3.8 environment.

Install the needed python packages:

    pip install -r requirements.txt

## Preparations

### Download the wiki2vec model:

Please download the model enwiki_20180420_100d.pkl.bz2 from [Github](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2) and extract the pkl file to the /model directory.

### Download a spacy language model:

    python -m spacy download en_core_web_trf

### Overview

In the following you find short explanations for the contents of each directory:

 - data:
   - vcslam: all needed files, extracted from the vcslam corpus
     - descriptions: textual descriptions from all concepts that are available in the ontology
     - history: historic mappings performed in the past: file history_0001.json holds the mappings for all data sources but 0001
     - images: simplified visualizations of the semantic models from vcslam
     - mappings: mappings between labels from the data sources and concepts from the ontology
     - models: all semantic models of vcslam including mappings and data documentations
 - models:
   - wiki2vec embeddings
 - pipeline: the actual DocSemMap pipeline


### Execution

To execute the pipeline, run the following command from the pipeline directory:

    python main.py

To adjust the data sources to be evaluated in the pipeline, just step into the main method of main.py