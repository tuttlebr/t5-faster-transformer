#!/bin/bash
python3 -m pip install --upgrade pip

pip3 install -q jupyterlab jaxlib==0.3.10 jax==0.3.13 transformers==4.19.2

# jupyter lab \
#     --ServerApp.ip=0.0.0.0 \
#     --ServerApp.port=8888 \
#     --ServerApp.allow_root=True \
#     --ServerApp.token='' \
#     --ServerApp.password='' \
#     --Application.log_level='CRITICAL'
./00_clone_fastertransformer_backend.sh
./01_clone_FasterTransformer_library.sh
./02_download_t5_weights.sh
./03_kernel_autotuning_t5.sh
# ./04_start_triton_inference_server.sh