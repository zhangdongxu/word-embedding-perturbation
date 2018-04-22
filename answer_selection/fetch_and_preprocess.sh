#!/bin/bash
# make sure the python version >=2.7, otherwise unzip
# the downloaded word embedding will fail
python2 scripts/download.py

glove_dir="data/glove"
glove_pre="glove.840B"
glove_dim="300d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
    python2 scripts/convert-wordvecs.py
fi
