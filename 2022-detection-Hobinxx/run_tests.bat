call conda env create -f environment.yml
call conda activate 5XSK0
call python -m pytest
PAUSE
call conda deactivate
