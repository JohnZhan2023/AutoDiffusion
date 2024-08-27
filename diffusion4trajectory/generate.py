from diffusion4trajectory import diffusion4trajectory, DiffusionConfig
from runner import load_dataset
from dataloader.nuplan_raster import nuplan_rasterize_collate_func_raw

import os
import pickle
import torch
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets.arrow_dataset import Dataset
def main():
    # set the path to the checkpoint and the root directory
    checkpoint = "/cephfs/zhanjh/diffusion4trajectory/checkpoint-59400"
    root = '/cephfs/shared/nuplan/online_s6/index'
    split = 'val'
    batchsize = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize the model
    config = DiffusionConfig.from_pretrained(checkpoint)
    diffusion = diffusion4trajectory.from_pretrained(checkpoint, config=config)
    diffusion = diffusion.to(device)
    
    all_maps_dic = {}
    map_folder = os.path.join(root[:-6], 'map')
    for each_map in os.listdir(map_folder):
        if each_map.endswith('.pkl'):
            map_path = os.path.join(map_folder, each_map)
            with open(map_path, 'rb') as f:
                map_dic = pickle.load(f)
            map_name = each_map.split('.')[0]
            all_maps_dic[map_name] = map_dic
            
    dataset = load_dataset(root, split, debug = True)
    
    # set the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        collate_fn=partial(
            nuplan_rasterize_collate_func_raw,
            all_maps_dic=all_maps_dic,
            dic_path=root[:-6],
            save_index=True,
        )
    )
    
    
    # make a mask
    mask = torch.ones(1,100,7)
    mask[:,:5] = 0
    mask[:,-20:] = 0 
    mask[:,18:21] = 0
    mask = mask.to(device)
    
    # generate the trajectory in a loop
    augmented_db = []
    file_num = 0
    for i, data in enumerate(dataloader):
        print(f"Processing batch {i}...")
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        data['mask'] = mask
        data["raw_trajectory_label"], data["valid_augmentation"] = augment_current_location(data["raw_trajectory_label"], data["scenario_type"], 20, offset_distance=1.0)
        output = diffusion.SDEdit_sample(data=data)
        augmented_trajectory = output["traj_logits"]
        # due to arrow's attribute, data should be of one dimention
        index = data['index']
        for i in range(len(index)):
            index[i]["augmented_trajectory"] = augmented_trajectory[i]
        augmented_db.extend(index)
        if i >30:
            break
    dataset = Dataset.from_list(augmented_db)
    dataset.save_to_disk(f"data/augmented_index")
    augmented_db = []


def convert_list_of_dicts_to_dict(data):
    """
    Converts a list of dictionaries to a dictionary where keys are column names
    and values are lists of column data.
    """
    if not data:
        raise ValueError("Data is empty")
    
    # Extract column names from the first dictionary
    columns = {key: [] for key in data[0].keys()}
    
    # Collect data for each column
    for record in data:
        for key in columns.keys():
            value = record.get(key, None)
            if isinstance(value, torch.Tensor):
                # Convert tensor to list
                columns[key].append(value.tolist())
            else:
                columns[key].append(value)

    return columns

def save_as_arrow(data, file_name):
    """
    Converts a list of dictionaries to a PyArrow Table and saves it as an Arrow file.
    """
    # Convert list of dictionaries to a dictionary of columns
    dict_data = convert_list_of_dicts_to_dict(data)
    
    # Create a PyArrow Table from the dictionary of columns
    table = pa.Table.from_pydict(dict_data)
    
    # Write the table to an Arrow file
    with pa.OSFile(file_name, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)



        
def augment_current_location(trajectory, scenario_type, step, offset_distance=0.5):
    device = trajectory.device  
    step -= 1
    
    # 确保所有操作都在 PyTorch 中完成
    prev_point = trajectory[:, step - 1, :2]  # xy of the previous step
    next_point = trajectory[:, step + 1, :2]  # xy of the next step
    current_point = trajectory[:, step-1:step+2, :2]  # xy of the current step
    
    tangent_vector = next_point - prev_point
    norm = torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
    valid = norm >= 1.0
    tangent_vector /= (norm + 1e-6)  # normalize with small epsilon to avoid division by zero
    
    # 直接在 PyTorch 中计算法线向量
    normal_vector = torch.stack([-tangent_vector[:, 1], tangent_vector[:, 0]], dim=1)
    offset = normal_vector * offset_distance
    offset = offset.unsqueeze(1).expand(-1, 3, -1)
    new_point = current_point + offset
    trajectory[:, step-1:step+2, :2] = new_point
    
    return trajectory, valid

        

if __name__ == "__main__":
    main()