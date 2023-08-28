# coding=utf-8

"""
Train a Transformer ML Model for Planning
"""

import logging
import os
import sys
import pickle
import copy
import torch
from tqdm import tqdm
import copy
import json
import datasets
import numpy as np
import evaluate
import transformers
from datasets import Dataset
from datasets.arrow_dataset import _concatenate_map_style_datasets
from functools import partial

from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformer4planning.models.model import build_models
from transformer4planning.utils import (
    ModelArguments, 
    DataTrainingArguments, 
    ConfigArguments, 
    PlanningTrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from transformer4planning.trainer import (PlanningTrainer, CustomCallback)
from torch.utils.data import DataLoader
from transformers.trainer_callback import DefaultFlowCallback

from datasets import Dataset, Value

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)

def load_dataset(root, split='train', dataset_scale=1, select=False):
    datasets = []
    index_root_folders = os.path.join(root, split)
    indices = os.listdir(index_root_folders)
 
    for index in indices:
        index_path = os.path.join(index_root_folders, index)
        if os.path.isdir(index_path):
            # load training dataset
            logger.info("Loading training dataset {}".format(index_path))
            dataset = Dataset.load_from_disk(index_path)
            if dataset is not None:
                datasets.append(dataset)
        else:
            continue
    # For nuplan dataset directory structure, each split obtains multi cities directories, so concat is required;
    # But for waymo dataset, index directory is just the datset, so load directory directly to build dataset. 
    if len(datasets) > 0: 
        dataset = _concatenate_map_style_datasets(datasets)
    else: 
        dataset = Dataset.load_from_disk(index_root_folders)
    # add split column
    dataset.features.update({'split': Value('string')})
    dataset = dataset.add_column(name='split', column=[split] * len(dataset))
    dataset.set_format(type='torch')
    if select:
        samples = int(len(dataset) * float(dataset_scale))
        dataset = dataset.select(range(samples))
    return dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConfigArguments, PlanningTrainingArguments))
    model_args, data_args, _, training_args = parser.parse_args_into_dataclasses()

    # pre-compute raster channels number
    if model_args.raster_channels == 0:
        road_types = 20
        agent_types = 8
        traffic_types = 4
        past_sample_number = int(2 * 20 / model_args.past_sample_interval)  # past_seconds-2, frame_rate-20
        if 'auto' not in model_args.model_name:
            # will cast into each frame
            if model_args.with_traffic_light:
                model_args.raster_channels = 1 + road_types + traffic_types + agent_types
            else:
                model_args.raster_channels = 1 + road_types + agent_types


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Handle config loading and saving
    # if config_args.load_model_config_from_path is not None:
    #     # Load the data class object from the JSON file
    #     model_parser = HfArgumentParser(ModelArguments)
    #     model_args, = model_parser.parse_json_file(config_args.load_model_config_from_path, allow_extra_keys=True)
    #     print(model_args)
    #     logger.warning("Loading model args, this will overwrite model args from command lines!!!")
    # if config_args.load_data_config_from_path is not None:
    #     # Load the data class object from the JSON file
    #     data_parser = HfArgumentParser(DataTrainingArguments)
    #     data_args, = data_parser.parse_json_file(config_args.load_data_config_from_path, allow_extra_keys=True)
    #     logger.warning("Loading data args, this will overwrite data args from command lines!!!")
    # if config_args.save_model_config_to_path is not None:
    #     with open(config_args.save_model_config_to_path, 'w') as f:
    #         json.dump(model_args.__dict__, f, indent=4)
    # if config_args.save_data_config_to_path is not None:
    #     with open(config_args.save_data_config_to_path, 'w') as f:
    #         json.dump(data_args.__dict__, f, indent=4)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Pass in the directory to load a saved dataset
    # See generation.py to process and save a dataset from the NuPlan Dataset
    """
    Set saved dataset folder to load a saved dataset
    1. Pass None to load from data_args.saved_dataset_folder as the root folder path to load all sub-datasets of each city
    2. Pass the folder of an index files to load one sub-dataset of one city
    """

    from datasets import disable_caching
    disable_caching()
    # loop all datasets
    logger.info("Loading full set of datasets from {}".format(data_args.saved_dataset_folder))
    assert os.path.isdir(data_args.saved_dataset_folder)
    if model_args.task == "nuplan" or model_args.task == "waymo": # nuplan and waymo datasets are stored in index format
        index_root = os.path.join(data_args.saved_dataset_folder, 'index')
    elif "diffusion" in model_args.task:
        index_root = data_args.saved_dataset_folder
    root_folders = os.listdir(index_root)
        
    if 'train' in root_folders:
        train_dataset = load_dataset(index_root, "train", data_args.dataset_scale, True)
    else:
        raise ValueError("No training dataset found in {}, must include at least one city in /train".format(index_root))
    
    if training_args.do_eval and 'test' in root_folders:
        test_dataset = load_dataset(index_root, "test", data_args.dataset_scale, False)
    else:
        print('Testset not found, using training set as test set')
        test_dataset = train_dataset
    
    if (training_args.do_eval or training_args.do_predict) and 'val' in root_folders:
        val_dataset = load_dataset(index_root, "val", data_args.dataset_scale, False)
    else:
        print('Validation set not found, using training set as val set')
        val_dataset = train_dataset

    if model_args.task == "nuplan":
        all_maps_dic = {}
        map_folder = os.path.join(data_args.saved_dataset_folder, 'map')
        for each_map in os.listdir(map_folder):
            if each_map.endswith('.pkl'):
                map_path = os.path.join(map_folder, each_map)
                with open(map_path, 'rb') as f:
                    map_dic = pickle.load(f)
                map_name = each_map.split('.')[0]
                all_maps_dic[map_name] = map_dic

    # loop split info and update for test set
    print('TrainingSet: ', train_dataset, '\nValidationSet', val_dataset, '\nTestingSet', test_dataset)

    dataset_dict = dict(
        train=train_dataset.shuffle(seed=training_args.seed),
        validation=val_dataset.shuffle(seed=training_args.seed),
        test=test_dataset.shuffle(seed=training_args.seed),
    )

    # Load a model's pretrained weights from a path or from hugging face's model base
    model = build_models(model_args)
    clf_metrics = dict(
        accuracy=evaluate.load("accuracy"),
        f1=evaluate.load("f1"),
        precision=evaluate.load("precision"),
        recall=evaluate.load("recall")
    )
    if 'auto' in model_args.model_name and model_args.k == -1:  # for the case action label as token 
        model.clf_metrics = clf_metrics
    elif model_args.next_token_scorer:
        assert model_args.k > 1 and model_args.ar_future_interval > 0, "ar_future_interval must be greater than 0 and k must be greater than 1"
        model.clf_metrics = clf_metrics

    if training_args.do_train:
        import multiprocessing
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(int(multiprocessing.cpu_count() / 8))
        train_dataset = dataset_dict["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = dataset_dict["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        predict_dataset = dataset_dict["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Initialize our Trainer
    if model_args.task == "nuplan":
        if model_args.encoder_type == "raster":
            from transformer4planning.preprocess.nuplan_rasterize import nuplan_rasterize_collate_func
            collate_fn = partial(nuplan_rasterize_collate_func,
                                dic_path=data_args.saved_dataset_folder,
                                all_maps_dic=all_maps_dic,
                                **model_args.__dict__)
        elif model_args.encoder_type == "vector":
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
            from transformer4planning.preprocess.pdm_vectorize import nuplan_vector_collate_func
            map_api = dict()
            for map in ['sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']:
                map_api[map] = get_maps_api(map_root=data_args.nuplan_map_path,
                                    map_version="nuplan-maps-v1.0",
                                    map_name=map
                                    )
            collate_fn = partial(nuplan_vector_collate_func, 
                                 dic_path=data_args.saved_dataset_folder, 
                                 map_api=map_api,
                                 use_centerline=model_args.use_centerline)
    elif model_args.task == "waymo":
        from transformer4planning.preprocess.waymo_vectorize import waymo_collate_func
        if model_args.encoder_type == "vector":
            collate_fn = partial(waymo_collate_func, 
                                 data_path=data_args.saved_dataset_folder, 
                                 interaction=model_args.interaction)
        elif model_args.encoder_type == "raster":
            raise NotImplementedError
    elif "diffusion" in model_args.task:
        from torch.utils.data._utils.collate import default_collate
        def feat_collate_func(batch, predict_yaw):
            excepted_keys = ['label', 'hidden_state']
            keys = batch[0].keys()
            result = dict()
            for key in keys:
                list_of_dvalues = []
                for d in batch:
                    if key in excepted_keys:
                        if key == "label" and not predict_yaw:
                            d[key] = d[key][:, :2]
                        list_of_dvalues.append(d[key])
                result[key] = default_collate(list_of_dvalues)
            return result
        collate_fn = partial(feat_collate_func, predict_yaw=model_args.predict_yaw)
    else:
        raise AttributeError("task must be nuplan or waymo or diffusion_decoder")

    trainer = PlanningTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        callbacks=[CustomCallback,],
        data_collator=collate_fn
    )
    trainer.pop_callback(DefaultFlowCallback)
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        result = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        logger.info("***** Final Eval results *****")
        logger.info(f"  {result}")
        # hyperparams = {"model": model_args.model_name, "dataset": data_args.saved_dataset_folder, "seed": training_args.seed}
        # evaluate.save("./results/", ** result, ** hyperparams)
        # logger.info(f" fde: {trainer.fde} ade: {trainer.ade}")

    if training_args.do_predict:
        # Currently only supports single GPU predict outputs
        """
        Will save prediction results, and dagger results if dagger is enabled
        """
        # TODO: fit new online process pipeline to save dagger and prediction results
        logger.info("*** Predict ***")
        with torch.no_grad():
            dagger_results = {
                'file_name':[],
                'frame_id':[],
                'rank':[],
                'ADE':[],
                'FDE':[],
                'y_bias':[]
            }
            prediction_results = {
                'file_names': [],
                'current_frame': [],
                'next_step_action': [],
                'predicted_trajectory': [],
            }
            test_dataloader = DataLoader(
                dataset=predict_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                num_workers=training_args.per_device_eval_batch_size,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True
            )


            if model_args.predict_trajectory:
                end_bias_x = []
                end_bias_y = []
                all_bias_x = []
                all_bias_y = []
                losses = []
                loss_fn = torch.nn.MSELoss(reduction="mean")

            for itr, input in enumerate(tqdm(test_dataloader)):
                # move batch to device
                for each_key in input:
                    if isinstance(input[each_key], type(torch.tensor(0))):
                        input[each_key] = input[each_key].to("cuda")

                eval_batch_size = training_args.per_device_eval_batch_size
                if model_args.autoregressive or model_args.ar_future_interval > 0:
                    # Todo: add autoregressive predict
                    traj_pred = model.generate(**input)
                else:
                    output = model(**copy.deepcopy(input))
                    traj_pred = output.logits                   
                    try:
                        file_name = input['file_name']
                        current_frame_idx = input['frame_id']
                    except:
                        file_name = ["null"] * eval_batch_size
                        current_frame_idx = -1 * torch.ones(eval_batch_size)
                    prediction_results['file_names'].extend(file_name)
                    prediction_results['current_frame'].extend(current_frame_idx.cpu().numpy())
                    if data_args.dagger:
                        dagger_results['file_name'].extend(file_name)
                        dagger_results['frame_id'].extend(list(current_frame_idx.cpu().numpy()))
                
                if model_args.predict_trajectory:
                    if model_args.autoregressive:# trajectory label as token case
                        trajectory_label = model.compute_normalized_points(input["trajectory"][:, 10:, :])
                        traj_pred = model.compute_normalized_points(traj_pred)
                        
                    else:
                        if 'mmtransformer' in model_args.model_name and model_args.task == 'waymo':
                            trajectory_label = input["trajectory_label"][:, :, :2]
                            trajectory_label = torch.where(trajectory_label != -1, trajectory_label, traj_pred)
                        else:
                            trajectory_label = input["trajectory_label"][:, 1::2, :]

                    loss = loss_fn(trajectory_label[:, :, :2], traj_pred[:, -trajectory_label.shape[1]:, :2])
                    end_trajectory_label = trajectory_label[:, -1, :]
                    end_point = traj_pred[:, -1, :]
                    end_bias_x.append(end_trajectory_label[:, 0] - end_point[:, 0])
                    end_bias_y.append(end_trajectory_label[:, 1] - end_point[:, 1])
                    all_bias_x.append(trajectory_label[:, :, 0] - traj_pred[:, -trajectory_label.shape[1]:, 0])
                    all_bias_y.append(trajectory_label[:, :, 1] - traj_pred[:, -trajectory_label.shape[1]:, 1])
                    losses.append(loss)

            if model_args.predict_trajectory:
                end_bias_x = torch.stack(end_bias_x, 0).cpu().numpy()
                end_bias_y = torch.stack(end_bias_y, 0).cpu().numpy()
                all_bias_x = torch.stack(all_bias_x, 0).reshape(-1).cpu().numpy()
                all_bias_y = torch.stack(all_bias_y, 0).reshape(-1).cpu().numpy()
                final_loss = torch.mean(torch.stack(losses, 0)).item()
                print('Mean L2 loss: ', final_loss)
                print('End point x offset: ', np.average(np.abs(end_bias_x)))
                print('End point y offset: ', np.average(np.abs(end_bias_y)))
                distance_error = np.sqrt(np.abs(all_bias_x)**2 + np.abs(all_bias_y)**2).reshape(-1, 80)
                final_distance_error = np.sqrt(np.abs(end_bias_x)**2 + np.abs(end_bias_y)**2)
                if data_args.dagger:
                    dagger_results['ADE'].extend(list(np.average(distance_error, axis=1).reshape(-1)))
                    dagger_results['FDE'].extend(list(final_distance_error.reshape(-1)))
                    dagger_results['y_bias'].extend(list(np.average(all_bias_y.reshape(-1, 80), axis=1).reshape(-1)))
                print('ADE', np.average(distance_error))
                print('FDE', np.average(final_distance_error))
            
            # print(dagger_results)
            def compute_dagger_dict(dic):
                tuple_list = list()
                fde_result_list = dict()
                y_bias_result_list = dict()
                for filename, id, ade, fde, y_bias in zip(dic["file_name"], dic["frame_id"], dic["ADE"], dic["FDE"], dic["y_bias"]):
                    if filename == "null":
                        continue
                    tuple_list.append((filename, id, ade, fde, abs(y_bias)))
    
                fde_sorted_list = sorted(tuple_list, key=lambda x:x[3], reverse=True)
                for idx, tp in enumerate(fde_sorted_list): 
                    if tp[0] in fde_result_list.keys():
                        fde_result_list[tp[0]]["frame_id"].append(tp[1])
                        fde_result_list[tp[0]]["ade"].append(tp[2])
                        fde_result_list[tp[0]]["fde"].append(tp[3])
                        fde_result_list[tp[0]]["y_bias"].append(tp[4])
                        fde_result_list[tp[0]]["rank"].append((idx+1)/len(fde_sorted_list))
                        
                    else:
                        fde_result_list[tp[0]] = dict(
                            frame_id=[tp[1]], ade=[tp[2]], fde=[tp[3]], y_bias=[tp[4]], rank=[(idx+1)/len(fde_sorted_list)]
                        )
                y_bias_sorted_list = sorted(tuple_list, key=lambda x:x[-1], reverse=True)
                for idx, tp in enumerate(y_bias_sorted_list): 
                    if tp[0] in y_bias_result_list.keys():
                        y_bias_result_list[tp[0]]["frame_id"].append(tp[1])
                        y_bias_result_list[tp[0]]["ade"].append(tp[2])
                        y_bias_result_list[tp[0]]["fde"].append(tp[3])
                        y_bias_result_list[tp[0]]["y_bias"].append(tp[4])
                        y_bias_result_list[tp[0]]["rank"].append((idx+1)/len(y_bias_sorted_list))
                    else:
                        y_bias_result_list[tp[0]] = dict(
                            frame_id=[tp[1]], ade=[tp[2]], fde=[tp[3]], y_bias=[tp[4]], rank=[(idx+1)/len(y_bias_sorted_list)]
                        )
                return fde_result_list, y_bias_result_list
            
            def draw_histogram_graph(data, title, savepath):
                import matplotlib.pyplot as plt
                plt.hist(data, bins=range(20), edgecolor='black')
                plt.title(title)
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(savepath, "{}.png".format(title)))
            if data_args.dagger:
                draw_histogram_graph(dagger_results["FDE"], title="FDE-distributions", savepath=training_args.output_dir)
                draw_histogram_graph(dagger_results["ADE"], title="ADE-distributions", savepath=training_args.output_dir)
                draw_histogram_graph(dagger_results["y_bias"], title="ybias-distribution", savepath=training_args.output_dir)
                fde_dagger_dic, y_bias_dagger_dic = compute_dagger_dict(dagger_results)


            if training_args.output_dir is not None:
                # save results
                output_file_path = os.path.join(training_args.output_dir, 'generated_predictions.pickle')
                with open(output_file_path, 'wb') as handle:
                    pickle.dump(prediction_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if data_args.dagger:
                    dagger_result_path = os.path.join(training_args.output_dir, "fde_dagger.pkl")
                    with open(dagger_result_path, 'wb') as handle:
                        pickle.dump(fde_dagger_dic, handle)
                    dagger_result_path = os.path.join(training_args.output_dir, "ybias_dagger.pkl")
                    with open(dagger_result_path, 'wb') as handle:
                        pickle.dump(y_bias_dagger_dic, handle)
                    print("dagger results save to {}".format(dagger_result_path))

        # predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        # metrics = predict_results.metrics
        # max_predict_samples = (
        #     data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        # )
        # metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        # trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)

        # if trainer.is_world_process_zero():
        #     if training_args.predict_with_generate:
        #         predictions = tokenizer.batch_decode(
        #             predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        #         )
        #         predictions = [pred.strip() for pred in predictions]
        #         output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
        #         with open(output_prediction_file, "w") as writer:
        #             writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_pretrain_name_or_path, "tasks": "NuPlanPlanning"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # Automatically saving all args into a json file.
    # TODO: Add this into Trainer class to save config while saving other logs
    # all_args_dic = {**model_args.__dict__, **data_args.__dict__, **config_args.__dict__, **training_args.__dict__}
    # if training_args.do_train:
    #     with open(os.path.join(training_args.output_dir, "training_args.json"), 'w') as f:
    #         json.dump(all_args_dic, f, indent=4)
    # elif training_args.do_eval:
    #     with open(os.path.join(training_args.output_dir, "eval_args.json"), 'w') as f:
    #         json.dump(all_args_dic, f, indent=4)

    return results


if __name__ == "__main__":
    main()
