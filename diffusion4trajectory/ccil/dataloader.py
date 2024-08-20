from runner import load_dataset
import os
import pickle
from functools import partial
from torch.utils.data import DataLoader


def load_data4AD(index_root, dataset_path, split="val14_1k", batch_size=1):
    # 导入你的数据集加载函数
    dataset = load_dataset(index_root, 1, split, 1, "all", False)

    # 加载地图数据
    all_maps_dic = {}
    map_folder = os.path.join(dataset_path, 'map')
    for each_map in os.listdir(map_folder):
        if each_map.endswith('.pkl'):
            map_path = os.path.join(map_folder, each_map)
            with open(map_path, 'rb') as f:
                map_dic = pickle.load(f)
            map_name = each_map.split('.')[0]
            all_maps_dic[map_name] = map_dic
    

    from diffusion4trajectory.dataloader.nuplan_raster import nuplan_rasterize_collate_func_raw
    collate_fn = partial(nuplan_rasterize_collate_func_raw,
                         dic_path=dataset_path,
                         all_maps_dic=all_maps_dic)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader