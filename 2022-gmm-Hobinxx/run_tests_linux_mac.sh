#!/usr/bin/env bash -l
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "${DIR}"
conda env create -f environment.yml
conda activate 5XSK0
python -m pytest
read -p "Press any key to start exit."
conda deactivate
