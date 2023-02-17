#!/bin/bash
clear
docker compose down
docker system prune -f
sudo rm -rf workspace/FasterTransformer 
sudo rm -rf workspace/fastertransformer_backend
sudo rm -rf workspace/t5-3b
docker compose build