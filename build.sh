#!/bin/bash

# Download GloVe embeddings
echo "Downloading GloVe embeddings..."
curl -O http://nlp.stanford.edu/data/glove.6B.zip

# Unzip the GloVe file into a 'glove' directory
echo "Unzipping GloVe embeddings..."
mkdir -p glove
unzip glove.6B.zip -d glove

# Remove the zip file to save space
rm glove.6B.zip
