python dagger_ccil/train_dynamics_model_for_AD.py --config=dagger_ccil/self_driving.yaml


python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    /cephfs/zhanjh/STR/StateTransformer_ccil/dagger_ccil/train_dynamics_model_for_AD.py \
    --config=dagger_ccil/self_driving.yaml


export CUDA_VISIBLE_DEVICES=0; torchrun --nproc_per_node=1 --nnodes=1 /cephfs/zhanjh/STR/StateTransformer_ccil/dagger_ccil/train_dynamics_model_for_AD.py --config=dagger_ccil/self_driving.yaml


python /cephfs/zhanjh/STR/StateTransformer_ccil/dagger_ccil/gen_aug_label_for_AD.py --config_path dagger_ccil/self_driving.yaml --test