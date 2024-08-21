# 80
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export WANDB_DISABLED=False; python -m torch.distributed.run --nproc_per_node=8 --master_port 12345 runner.py --model_name scratch-mixtral-small-diffusion-refiner --model_pretrain_name_or_path /cephfs/zhanjh/exp/MOE_aux/checkpoint-150000 --saved_dataset_folder /localssd/zhanjh/online_s6 --output_dir /cephfs/zhanjh/StrDiff_result/DiffusionRefiner_80 --logging_dir /cephfs/zhanjh/ExplicitDiffusion_logging --run_name DiffusionRefiner_80 --num_train_epochs 50 --per_device_train_batch_size 64 --warmup_steps 50 --weight_decay 0.01 --logging_steps 100 --save_strategy steps --save_steps 300 --dataloader_num_workers 24 --dataloader_drop_last True --save_total_limit 10 --do_train --task nuplan --remove_unused_columns False --do_eval --evaluation_strategy steps --eval_steps 100 --per_device_eval_batch_size 8 --predict_yaw True --use_proposal 0 --selected_exponential_past True --mean_circular_loss True --raster_channels 34 --use_mission_goal False --raster_encoder_type vit --vit_intermediate_size 768 --lr_scheduler_type cosine --use_speed --use_key_points specified_backward --augment_index 5 --attn_implementation flash_attention_2 --sync_norm True --bf16 True --nuplan_sim_exp_root /localssd/zhanjh/online_s6 --nuplan_sim_data_path /localssd/zhanjh/online_s6/val --nuplan_sim_map_folder /localssd/zhanjh/online_s6/map --nuplan_sim_split_filter_yaml nuplan_simulation/val14_split.yaml --max_sim_samples 64 --inspect_kp_loss --num_local_experts 24 --num_experts_per_token 2 --router_aux_loss_coef 0.1 --overwrite_output_dir  --ddp_find_unused_parameters False --debug_train True --eval_on_start True  --max_eval_samples 64 
 
# 10
STACKS=80 #80
WANDB_DISABLED=True
OBJECTIVE=pred_x0 # pred_x0 pred_noise
SCHEDULER=sigmoid #  cosine
DATA=/localssd/zhanjh/nuplan/online_s6/ # /cephfs/shared/nuplan/online_s6/  /localssd/zhanjh/online_s6/
DDIM=False
DIFFUSION_TIMES=10
MAP_CONDITION=True
NORMALIZE=True
SCALE=25
RESIDUAL=True
AUG=True
STAMP=TMP



export CUDA_VISIBLE_DEVICES=0;export WANDB_DISABLED=${WANDB_DISABLED};python -m torch.distributed.run --nproc_per_node=1 --master_port 12345 runner.py --model_name scratch-mixtral-small-refiner --backbone_path /cephfs/zhanjh/exp/MOE_aux/checkpoint-150000 --saved_dataset_folder ${DATA} --output_dir /cephfs/zhanjh/ExplicitDiffusion_${STACKS}_${OBJECTIVE}_${SCHEDULER}_${NORMALIZE}_${SCALE}_res_${RESIDUAL}_aug_${AUG}_${STAMP} --logging_dir /cephfs/zhanjh/DiffusionRefiner_${STACKS}_${OBJECTIVE}_${SCHEDULER}_${NORMALIZE}_${SCALE}_res_${RESIDUAL}_aug_${AUG}_${STAMP} --run_name DiffusionRefiner_${STACKS}_${OBJECTIVE}_${SCHEDULER}_${NORMALIZE}_${SCALE}_res_${RESIDUAL}_aug_${AUG}_${STAMP} --num_train_epochs 50 --per_device_train_batch_size 64 --warmup_steps 50 --weight_decay 0.01 --logging_steps 100 --save_strategy steps --save_steps 3000 --dataloader_num_workers 24 --dataloader_drop_last True --save_total_limit 5 --do_train --task nuplan --remove_unused_columns False --do_eval --evaluation_strategy steps --eval_steps 1000 --per_device_eval_batch_size 8 --predict_yaw True --use_proposal 0 --selected_exponential_past True --mean_circular_loss True --raster_channels 34 --use_mission_goal False --raster_encoder_type vit --vit_intermediate_size 768 --lr_scheduler_type cosine --use_speed --use_key_points specified_backward --augment_index 5 --attn_implementation flash_attention_2 --sync_norm True --bf16 True --nuplan_sim_exp_root ${DATA} --nuplan_sim_data_path ${DATA}map --nuplan_sim_split_filter_yaml nuplan_simulation/val14_split.yaml --max_sim_samples 64 --inspect_kp_loss --num_local_experts 24 --num_experts_per_token 2 --router_aux_loss_coef 0.1 --overwrite_output_dir  --ddp_find_unused_parameters False --debug_train True --eval_on_start False --max_eval_samples 64 --ddim ${DDIM} --stacks ${STACKS} --objective ${OBJECTIVE} --beta_schedule ${SCHEDULER} --diffusion_timesteps ${DIFFUSION_TIMES}  --map_cond ${MAP_CONDITION}  --normalize ${NORMALIZE} --learning_rate 0.0001 --residual ${RESIDUAL} --augment_current_pose_rate 0.3 --augment_current_with_past_linear_changes ${AUG} --augment_current_with_future_linear_changes ${AUG}  --trajectory_prediction_mode off_roadx100
# --resume_from_checkpoint /cephfs/zhanjh/ExplicitDiffusion_10_pred_x0_sigmoid_True_25_res_True_aug_True/checkpoint-36000

# Debug
# export CUDA_VISIBLE_DEVICES=0; python -m torch.distributed.run --nproc_per_node=1 --master_port 12346 runner.py --model_name scratch-mixtral-small-diffusion-dit --model_pretrain_name_or_path /cephfs/zhanjh/exp/MOE_aux/checkpoint-150000 --saved_dataset_folder /localssd/zhanjh/online_s6 --output_dir /cephfs/zhanjh/StrDiff_result/DiT --logging_dir /cephfs/zhanjh/StrDiff_result_2 --run_name Small_Str_DiT_2 --num_train_epochs 50 --per_device_train_batch_size 32 --warmup_steps 50 --weight_decay 0.01 --logging_steps 100 --save_strategy steps --save_steps 3000 --dataloader_num_workers 24 --dataloader_drop_last True --save_total_limit 10 --do_train --task nuplan --remove_unused_columns False --do_eval --evaluation_strategy steps --eval_steps 300 --per_device_eval_batch_size 1 --predict_yaw True --use_proposal 0 --selected_exponential_past True --mean_circular_loss True --raster_channels 34 --use_mission_goal False --raster_encoder_type vit --vit_intermediate_size 768 --lr_scheduler_type cosine --use_speed --use_key_points specified_backward --augment_index 5 --attn_implementation flash_attention_2 --sync_norm True --bf16 True --nuplan_sim_exp_root /localssd/zhanjh/online_s6 --nuplan_sim_data_path /localssd/zhanjh/online_s6/val --nuplan_sim_map_folder /localssd/zhanjh/online_s6/map --nuplan_sim_split_filter_yaml nuplan_simulation/val14_split.yaml --max_sim_samples 64 --inspect_kp_loss --num_local_experts 24 --num_experts_per_token 2 --router_aux_loss_coef 0.1 --overwrite_output_dir  --ddp_find_unused_parameters False --debug_train True --eval_on_start True  --max_eval_samples 10 
