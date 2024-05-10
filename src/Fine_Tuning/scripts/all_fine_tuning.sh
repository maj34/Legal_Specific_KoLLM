export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1

# supervised fine-tuning
python3.8 ../fine_tuning.py --config_file /workspace/Legal_Specific_KoLLM/src/LLM_Fine_Tuning/configs/eeve.yaml
python3.8 ../fine_tuning.py --config_file /workspace/Legal_Specific_KoLLM/src/LLM_Fine_Tuning/configs/eeve-instruct.yaml
python3.8 ../fine_tuning.py --config_file /workspace/Legal_Specific_KoLLM/src/LLM_Fine_Tuning/configs/kullm3.yaml
python3.8 ../fine_tuning.py --config_file /workspace/Legal_Specific_KoLLM/src/LLM_Fine_Tuning/configs/llama3-instruct.yaml

# further training -> supervised fine-tuning
python3.8 ../further_training_fine_tuning.py --config_file /workspace/Legal_Specific_KoLLM/src/LLM_Fine_Tuning/configs/eeve.yaml
