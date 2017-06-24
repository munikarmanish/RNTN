#!/bin/sh

# Dataset
echo ":: Downloading dataset..."
DATASET_ZIP=trainDevTestTrees_PTB.zip
wget -c https://nlp.stanford.edu/sentiment/${DATASET_ZIP}
echo ":: Extracting dataset into 'trees' folder..."
unzip ${DATASET_ZIP}
rm ${DATASET_ZIP}

# Create a folder for storing models
echo ":: Creating 'models' directory to store models..."
mkdir -p models
