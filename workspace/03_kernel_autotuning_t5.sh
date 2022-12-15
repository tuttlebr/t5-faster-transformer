#!/bin/bash

batch_size=1
beam_width=1
max_mem_seq_len=32
encoder_d_model=1024
encoder_head_num=32
encoder_size_per_head=128
encoder_inter_size=16384
decoder_d_model=1024
decoder_head_num=32
decoder_size_per_head=128
decoder_inter_size=16384
decoder_vocab_size=32128
data_type=1
tensor_para_size=1
is_fp16_compute_type=1
is_append=1

# -i: The path of megatron model
# -o: The output path of converted model
# -i_g: The tensor parallel size we hope for inference also the number of GPUs expected for inference.
# -weight_data_type: Data type of weights fp32 or fp16, should align with is_fp16_compute_type=1 arg for
# kernel optimization as well!
python3 FasterTransformer/examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -i t5-3b/ \
        -o ./models/t5-3b/ \
        -i_g ${tensor_para_size} \
        -weight_data_type fp16
        
CUDA_VISIBLE_DEVICES=0 ./FasterTransformer/build/bin/t5_gemm \
    ${batch_size} \
    ${beam_width} \
    ${max_mem_seq_len} \
    ${encoder_d_model} \
    ${encoder_head_num} \
    ${encoder_size_per_head} \
    ${encoder_inter_size} \
    ${decoder_d_model} \
    ${decoder_head_num} \
    ${decoder_size_per_head} \
    ${decoder_inter_size} \
    ${decoder_vocab_size} \
    ${data_type} \
    ${tensor_para_size} \
    ${is_fp16_compute_type} \
    ${is_append} 
    
mkdir -p triton-model-store
cp -r ./fastertransformer_backend/all_models/t5/ triton-model-store/

printf "%s\n" \
"" \
"We have to open the copied TRITON config for T5 model triton-model-store/t5/fastertransformer/config.pbtxt." \
"Only two mandatory parameters need to be changed there to start inference:" \
"We have to update tensor_para_size. We prepared our weights for ${tensor_para_size} GPUs, so we have to set it equal to ${tensor_para_size}." \
"" \
"parameters {" \
"  key: "tensor_para_size"" \
"  value: {" \
"    string_value: ${tensor_para_size}" \
"  }" \
"}" \
"" \
"We have to update the path to the checkpoint prepared for ${tensor_para_size}-GPU inference (folder from the previous step)" \
"parameters {" \
"  key: "model_checkpoint_path"" \
"  value: {" \
"    string_value: "/ft_workspace/models/t5-3b/${tensor_para_size}-gpu/""  \
"  }" \
"}"