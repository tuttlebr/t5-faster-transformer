#!/bin/bash
python3 -m pip install --upgrade pip

pip3 install -q jupyterlab jaxlib==0.3.10 jax==0.3.13 transformers==4.19.2 sentencepiece==0.1.97

jupyter lab \
    --ServerApp.ip=0.0.0.0 \
    --ServerApp.port=8888 \
    --ServerApp.allow_root=True \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --Application.log_level='CRITICAL'