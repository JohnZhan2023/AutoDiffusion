from runner import load_dataset
import hydra
import os
import pickle
from functools import partial
import torch
import logging
import numpy as np


from transformers import (
    HfArgumentParser,
    set_seed,
)
from diffusion4trajectory import diffusion4trajectory, DiffusionConfig
from transformer4planning.trainer import (PlanningTrainer, CustomCallback)
from transformer4planning.trainer import compute_metrics
from transformers.trainer_callback import DefaultFlowCallback
from transformer4planning.utils.args import (
    ModelArguments, 
    DataTrainingArguments, 
    ConfigArguments, 
    PlanningTrainingArguments
)
from transformer4planning.models.backbone.str_base import STRConfig
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConfigArguments, PlanningTrainingArguments))
    model_args, data_args, config_args, training_args = parser.parse_args_into_dataclasses()
    root = data_args.saved_dataset_folder + "index"
    debug = training_args.debug_train
    max_eval_samples = data_args.max_eval_samples
    
    train_dataset = load_dataset(root, "train", debug = debug)
    val_dataset = load_dataset(root, "val", debug = debug)
    
    all_maps_dic = {}
    map_folder = os.path.join(root[:-6], 'map')
    for each_map in os.listdir(map_folder):
        if each_map.endswith('.pkl'):
            map_path = os.path.join(map_folder, each_map)
            with open(map_path, 'rb') as f:
                map_dic = pickle.load(f)
            map_name = each_map.split('.')[0]
            all_maps_dic[map_name] = map_dic
    if training_args.model_pretrain_name_or_path is not None:
        config = DiffusionConfig.from_pretrained(training_args.model_pretrain_name_or_path)
        model = diffusion4trajectory(training_args.model_pretrain_name_or_path, config)
    else:
        config = DiffusionConfig()
        config.update_by_model_args(model_args)
        config.input_dim = 7
        config.output_dim = 7
        config.horizon = 100
        config.n_cond_steps = 788
        config.cond_dim = 256
        config.map_cond = True
        config.n_cond_layers = 4
        config.timesteps = 100
        config.objective = "pred_x0"
        config.beta_schedule = "sigmoid"
        config.use_proposal = False
        config.n_embd = 256
        config.n_head = 8
        config.n_layer = 4
        config.use_key_points = "specified_backward"
        

        model = diffusion4trajectory(config)
    
    num_samples = min(len(val_dataset), max_eval_samples)
    val_dataset = val_dataset.select(range(num_samples))
    from dataloader.nuplan_raster import nuplan_rasterize_collate_func_raw
    collate_fn = partial(nuplan_rasterize_collate_func_raw,
                            dic_path=root[:-6],
                            all_maps_dic=all_maps_dic,
                            )
    
    trainer = PlanningTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset if training_args.do_train else None,  # training dataset
        eval_dataset=val_dataset ,
        callbacks=[CustomCallback,],
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    
    
    # manage Megatron if set to use
    from accelerate import DistributedType
    if trainer.accelerator.distributed_type == DistributedType.MEGATRON_LM:
        from accelerate.utils import MegatronLMDummyScheduler
        lr_scheduler = MegatronLMDummyScheduler(
            optimizer=trainer.optimizer,
            total_num_steps=training_args.num_train_steps,
            warmup_num_steps=training_args.warmup_steps,
        )
        trainer.lr_scheduler = lr_scheduler
        from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
        from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
        from transformers.utils import (
            SAFE_WEIGHTS_NAME,
            WEIGHTS_NAME,
        )
        import safetensors
        TRAINING_ARGS_NAME = "training_args.bin"
        import types
        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

            supported_classes = (PreTrainedModel,)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model, supported_classes):
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if isinstance(unwrap_model(self.model), supported_classes):
                    trainer.accelerator.save_state(output_dir)
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                        )
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                trainer.accelerator.save_state(output_dir)

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        trainer.save_model = types.MethodType(_save, trainer)

    trainer.pop_callback(DefaultFlowCallback)
    
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()
    
if __name__ == "__main__":
    main()