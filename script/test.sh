python run_simulation.py --test_type open_loop_boxes --data_path /cephfs/shared/nuplan-v1.1/test --map_path /cephfs/shared/nuplan-v1.1/maps --model_path /cephfs/zhanjh/ExplicitDiffusion_80_pred_x0_sigmoid_True_25_res_True_aug_True/checkpoint-29400 --split_filter_yaml nuplan_simulation/test14_hard.yaml --max_scenario_num 10000 --batch_size 8 --device cuda --exp_folder Wednesday_test_whole_open_loop --processes-repetition 8