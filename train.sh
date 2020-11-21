#!/usr/bin/env bash
TASK_NAME=laptop14
ABSA_TYPE=Full-Hinge-Loss
CUDA_VISIBLE_DEVICES=0,2,3 python main.py --model_type bert \
                         --absa_type ${ABSA_TYPE} \
                         --tfm_mode finetune \
                         --fix_tfm 0 \
                         --model_name_or_path activebus/BERT-DK_laptop \
                         --data_dir ./data/${TASK_NAME} \
                         --task_name ${TASK_NAME} \
                         --per_gpu_train_batch_size 16 \
                         --per_gpu_eval_batch_size 8 \
                         --learning_rate 4e-5 \
                         --do_train \
                         --do_eval \
                         --evaluate_during_training \
                         --load_model laptop/alpha-0.5/saved_model-1000 \
                         --do_adv \
                         --adv_data_path adv-laptop/train.pth \
                         --adv_loss_weight 0.2 \
                         --do_lower_case \
                         --tagging_schema BIO \
                         --overfit 0 \
                         --eval_all_checkpoints \
                         --overwrite_output_dir \
                         --MASTER_ADDR localhost \
                         --MASTER_PORT 28512 \
                         --max_steps 2000 \
