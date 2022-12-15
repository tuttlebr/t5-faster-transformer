#!/bin/bash

git-lfs install
pip install -q -r fastertransformer_backend/tools/t5_utils/t5_requirement.txt
git clone https://huggingface.co/t5-3b