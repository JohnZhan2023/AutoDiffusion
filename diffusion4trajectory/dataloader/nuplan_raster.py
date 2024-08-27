import numpy as np
import pickle
import math, random
import cv2
import shapely.geometry
import os
import torch
from functools import partial

from torch.utils.data._utils.collate import default_collate
from transformer4planning.utils.nuplan_utils import generate_contour_pts, normalize_angle, change_coordination
from transformer4planning.utils import nuplan_utils
from transformer4planning.utils.common_utils import save_raster
from transformer4planning.preprocess.nuplan_rasterize import draw_rasters, load_data


def nuplan_rasterize_collate_func_augmentation(batch, dic_path=None, autoregressive=False, augmentation = "naive", **encode_kwargs):
    """
    'nuplan_collate_fn' for augmentation
    """
    # padding for tensor data
    expected_padding_keys = ["road_ids", "route_ids", "traffic_ids"]
    agent_id_lengths = list()
    for i, d in enumerate(batch):
        agent_id_lengths.append(len(d["agent_ids"]))
    max_agent_id_length = max(agent_id_lengths)
    for i, d in enumerate(batch):
        agent_ids = d["agent_ids"]
        agent_ids.extend(["null"] * (max_agent_id_length - len(agent_ids)))
        batch[i]["agent_ids"] = agent_ids

    padded_tensors = dict()
    for key in expected_padding_keys:
        tensors = [data[key] for data in batch]
        padded_tensors[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=-1)
        for i, _ in enumerate(batch):
            batch[i][key] = padded_tensors[key][i]


    map_func = partial(static_coor_rasterize_augmentation, data_path=dic_path, augmentation = augmentation, **encode_kwargs)

    new_batch = list()
    for i, d in enumerate(batch):
        rst = map_func(d)
        if rst is None:
            continue
        new_batch.append(rst)

    if len(new_batch) == 0:
        return {}
    
    result = dict()
    for key in new_batch[0].keys():
        if key is None:
            continue
        list_of_dvalues = []
        for d in new_batch:
            if d[key] is not None:
                list_of_dvalues.append(d[key])
            elif key == "scenario_type":
                list_of_dvalues.append('Unknown')
            else:
                print('Error: None value', key, d[key])   # scenario_type might be none for older dataset
        result[key] = default_collate(list_of_dvalues)
    return result


def static_coor_rasterize_augmentation(sample, data_path, augmentation, raster_shape=(224, 224),
                          frame_rate=20, past_seconds=2, future_seconds=8,
                          high_res_scale=4, low_res_scale=0.77,
                          road_types=20, agent_types=8, traffic_types=4,
                          past_sample_interval=2, future_sample_interval=2,
                          debug_raster_path=None, all_maps_dic=None, agent_dic=None,
                          frequency_change_rate=2,
                          **kwargs):
    """
    WARNING: frame_rate has been change to 10 as default to generate new dataset pickles, this is automatically processed by hard-coded logits
    :param sample: a dictionary containing the following keys:
        - file_name: the name of the file
        - map: the name of the map, ex: us-ma-boston
        - split: the split, train, val or test
        - road_ids: the ids of the road elements
        - agent_ids: the ids of the agents in string
        - traffic_ids: the ids of the traffic lights
        - traffic_status: the status of the traffic lights
        - route_ids: the ids of the routes
        - frame_id: the frame id of the current frame, this is the global index which is irrelevant to frame rate of agent_dic pickles (20Hz)
        - debug_raster_path: if a debug_path past, will save rasterized images to disk, warning: will slow down the process
    :param data_path: the root path to load pickle files
    starting_frame, ending_frame, sample_frame in 20Hz,
    """

    filename = sample["file_name"]
    map = sample["map"]
    split = sample["split"]

    if split == 'val14_1k':
        split = 'val'
    elif split == 'test_hard14_index':
        split = 'test'

    frame_id = sample["frame_id"]  # current frame of this sample
    data_dic = load_data(sample, data_path, all_maps_dic)
    
    
    
    agent_dic = data_dic["agent_dic"]
    y_inverse = data_dic["y_inverse"]

    assert agent_dic['ego']['starting_frame'] == 0, f'ego starting frame {agent_dic["ego"]["starting_frame"]} should be 0'

    # augment frame id
    augment_frame_id = kwargs.get('augment_index', 0)
    if augment_frame_id != 0 and 'train' in split:
        frame_id += random.randint(-augment_frame_id - 1, augment_frame_id)
        frame_id = max(frame_id, past_seconds * frame_rate)

    # if new version of data, using relative frame_id
    relative_frame_id = True if 'starting_frame' in agent_dic['ego'] else False

    if "train" in split and kwargs.get('augment_current_pose_rate', 0) > 0:
        # copy agent_dic before operating to it
        ego_pose_agent_dic = agent_dic['ego']['pose'].copy()
    else:
        ego_pose_agent_dic = agent_dic['ego']['pose']

    # calculate frames to sample
    scenario_start_frame = frame_id - past_seconds * frame_rate
    scenario_end_frame = frame_id + future_seconds * frame_rate

    # for example,
    if kwargs.get('selected_exponential_past', True):
        # 2s, 1s, 0.5s, 0s
        # sample_frames_in_past = [scenario_start_frame + 0, scenario_start_frame + 20, scenario_start_frame + 30]
        sample_frames_in_past = [scenario_start_frame + 0, scenario_start_frame + 20, scenario_start_frame + 30, frame_id]
    elif kwargs.get('current_frame_only', False):
        sample_frames_in_past = [frame_id]
    else:
        # [10, 11, ...., 10+(2+8)*20=210], past_interval=2, future_interval=2, current_frame=50
        # sample_frames_in_past = [10, 12, 14, ..., 48], number=(50-10)/2=20
        sample_frames_in_past = list(range(scenario_start_frame, frame_id, past_sample_interval))  # add current frame in the end
    # sample_frames_in_future = [52, 54, ..., 208, 210], number=(210-50)/2=80
    sample_frames_in_future = list(range(frame_id + future_sample_interval, scenario_end_frame + future_sample_interval, future_sample_interval))  # + one step to avoid the current frame

    sample_frames = sample_frames_in_past + sample_frames_in_future
    # sample_frames = list(range(scenario_start_frame, frame_id + 1, frame_sample_interval))

    # augment current position
    aug_current = 0
    aug_rate = kwargs.get('augment_current_pose_rate', 0)
    if "train" in split and aug_rate > 0 and random.random() < aug_rate:
        augment_current_ratio = kwargs.get('augment_current_pose_ratio', 0.3)
        augment_current_with_past_linear_changes = kwargs.get('augment_current_with_past_linear_changes', False)
        augment_current_with_future_linear_changes = kwargs.get('augment_current_with_future_linear_changes', False)
        speed_per_step = nuplan_utils.euclidean_distance(
            ego_pose_agent_dic[frame_id // frequency_change_rate, :2],
            ego_pose_agent_dic[frame_id // frequency_change_rate - 5, :2]) / 5.0
        aug_x = augment_current_ratio * speed_per_step
        aug_y = augment_current_ratio * speed_per_step
        yaw_noise_scale = 0.05  # 360 * 0.05 = 18 degree
        aug_yaw = (random.random() * 2 - 1) * yaw_noise_scale
        dx = (random.random() * 2 - 1) * aug_x
        dy = (random.random() * 2 - 1) * aug_y
        dyaw = (random.random() * 2 * np.pi - np.pi) * aug_yaw
        ego_pose_agent_dic[frame_id//frequency_change_rate, 0] += dx
        ego_pose_agent_dic[frame_id//frequency_change_rate, 1] += dy
        ego_pose_agent_dic[frame_id//frequency_change_rate, -1] += dyaw
        aug_current = 1
        if augment_current_with_future_linear_changes:
            # linearly project the past poses
            # generate a numpy array decaying from 1 to 0 with shape of 80, 4
            decay = np.ones((80, 4)) * np.linspace(1, 0, 80).reshape(-1, 1)
            decay[:, 0] *= dx
            decay[:, 1] *= dy
            decay[:, 2] *= 0
            decay[:, 3] *= dyaw
            ego_pose_agent_dic[frame_id // frequency_change_rate: frame_id // frequency_change_rate + 80, :] += decay

        if augment_current_with_past_linear_changes:
            # generate a numpy array raising from 0 to 1 with the shape of 20, 4
            raising = np.ones((20, 4)) * np.linspace(0, 1, 20).reshape(-1, 1)
            raising[:, 0] *= dx
            raising[:, 1] *= dy
            raising[:, 2] *= 0
            raising[:, 3] *= dyaw
            ego_pose_agent_dic[frame_id // frequency_change_rate - 21: frame_id // frequency_change_rate - 1, :] += raising
    if random.random() < 0.5:
        if augmentation == "naive":
            ego_pose_agent_dic = augment_and_smooth_trajectory(ego_pose_agent_dic, frame_id // frequency_change_rate, augment_x = 3.0, augment_y = 0.2, smooth_factor=0.85)
        elif augmentation == "diffusion":
            assert "diffusion_trajecotry" in sample.keys(), "diffusion trajectory not in sample keys"
            ego_pose_agent_dic = sample["diffusion_trajecotry"]
        elif augmentation == "ccil":
            NotImplemented

    
    # initialize rasters
    origin_ego_pose = ego_pose_agent_dic[frame_id//frequency_change_rate].copy()  # hard-coded resample rate 2
    if kwargs.get('skip_yaw_norm', False):
        origin_ego_pose[-1] = 0

    if "agent_ids" not in sample.keys():
        if 'agent_ids_index' in sample.keys():
            agent_ids = []
            all_agent_ids = list(agent_dic.keys())
            for each_agent_index in sample['agent_ids_index']:
                if each_agent_index == -1:
                    continue
                if each_agent_index > len(all_agent_ids):
                    print(f'Warning: agent_ids_index is larger than agent_dic {each_agent_index} {len(all_agent_ids)}')
                    continue
                agent_ids.append(all_agent_ids[each_agent_index])
            assert 'ego' in agent_ids, 'ego should be in agent_ids'
        else:
            assert False
        # print('Warning: agent_ids not in sample keys')
        # agent_ids = []
        # max_dis = 300
        # for each_agent in agent_dic:
        #     starting_frame = agent_dic[each_agent]['starting_frame']
        #     target_frame = frame_id - starting_frame
        #     if target_frame < 0 or frame_id >= agent_dic[each_agent]['ending_frame']:
        #         continue
        #     pose = agent_dic[each_agent]['pose'][target_frame//frequency_change_rate, :].copy()
        #     if pose[0] < 0 and pose[1] < 0:
        #         continue
        #     pose -= origin_ego_pose
        #     if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
        #         continue
        #     agent_ids.append(each_agent)
    else:
        agent_ids = sample["agent_ids"]  # list of strings

    # num_frame = torch.div(frame_id, frequency_change_rate, rounding_mode='floor')
    # origin_ego_pose = agent_dic["ego"]["pose"][num_frame].copy()  # hard-coded resample rate 2
    if np.isinf(origin_ego_pose[0]) or np.isinf(origin_ego_pose[1]):
        assert False, f"Error: ego pose is inf {origin_ego_pose}, not enough precision while generating dictionary"

    rasters_high_res, rasters_low_res = draw_rasters(
        data_dic, origin_ego_pose, agent_ids,
        road_types, traffic_types, agent_types,
        sample_frames_in_past, frequency_change_rate,
        autoregressive=False,
        raster_shape=raster_shape,
        high_res_scale=high_res_scale,
        low_res_scale=low_res_scale,
        **kwargs
    )

    # context action computation
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])
    context_actions = list()
    ego_poses = ego_pose_agent_dic - origin_ego_pose
    rotated_poses = np.array([ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                              ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_,
                              np.zeros(ego_poses.shape[0]), ego_poses[:, -1]]).transpose((1, 0))
    rotated_poses[:, 1] *= y_inverse

    if kwargs.get('use_speed', True):
        # speed, old data dic does not have speed key
        speed = agent_dic['ego']['speed']  # v, a, angular_v
        if speed.shape[0] == ego_poses.shape[0] * 2:
            speed = speed[::2, :]
        for i in sample_frames_in_past:
            selected_pose = rotated_poses[i // frequency_change_rate]  # hard-coded frequency change
            selected_pose[-1] = normalize_angle(selected_pose[-1])
            action = np.concatenate((selected_pose, speed[i // frequency_change_rate]))
            context_actions.append(action)
    else:
        for i in sample_frames_in_past:
            action = rotated_poses[i//frequency_change_rate]  # hard-coded frequency change
            action[-1] = normalize_angle(action[-1])
            context_actions.append(action)

    # future trajectory
    # check if samples in the future is beyond agent_dic['ego']['pose'] length
    if relative_frame_id:
        sample_frames_in_future = (np.array(sample_frames_in_future, dtype=int) - agent_dic['ego']['starting_frame']) // frequency_change_rate
    if sample_frames_in_future[-1] >= ego_pose_agent_dic.shape[0]:
        # print('sample index beyond length of agent_dic: ', sample_frames_in_future[-1], agent_dic['ego']['pose'].shape[0])
        return None

    result_to_return = dict()

    trajectory_label = ego_pose_agent_dic[sample_frames_in_future, :].copy()

    # get a planning trajectory from a CBC constant velocity planner
    # if kwargs.get('use_cbc_planner', False):
    #     from transformer4planning.rule_based_planner.nuplan_base_planner import MultiPathPlanner
    #     planner = MultiPathPlanner(road_dic=road_dic)
    #     planning_result = planner.plan_marginal_trajectories(
    #         my_current_pose=origin_ego_pose,
    #         my_current_v_mph=agent_dic['ego']['speed'][frame_id//frequency_change_rate, 0],
    #         route_in_blocks=sample['route_ids'].numpy().tolist(),
    #     )
    #     _, marginal_trajectories, _ = planning_result
    #     result_to_return['cbc_planning'] = marginal_trajectories
    trajectory_label -= origin_ego_pose
    traj_x = trajectory_label[:, 0].copy()
    traj_y = trajectory_label[:, 1].copy()
    trajectory_label[:, 0] = traj_x * cos_ - traj_y * sin_
    trajectory_label[:, 1] = traj_x * sin_ + traj_y * cos_
    trajectory_label[:, 1] *= y_inverse

    result_to_return["high_res_raster"] = np.array(rasters_high_res, dtype=bool)
    result_to_return["low_res_raster"] = np.array(rasters_low_res, dtype=bool)
    result_to_return["context_actions"] = np.array(context_actions, dtype=np.float32)
    result_to_return['trajectory_label'] = trajectory_label.astype(np.float32)

    del rasters_high_res
    del rasters_low_res
    del trajectory_label
    # print('inspect: ', result_to_return["context_actions"].shape)

    camera_image_encoder = kwargs.get('camera_image_encoder', None)
    if camera_image_encoder is not None and 'test' not in split:
        import PIL.Image
        # load images
        if 'train' in split:
            images_folder = kwargs.get('train_camera_image_folder', None)
        elif 'val' in split:
            images_folder = kwargs.get('val_camera_image_folder', None)
        else:
            raise ValueError('split not recognized: ', split)

        images_paths = sample['images_path']
        if images_folder is None or len(images_paths) == 0:
            print('images_folder or images_paths not valid', images_folder, images_paths, filename, map, split, frame_id)
            return None
        if len(images_paths) != 8:
            # clean duplicate cameras
            camera_dic = {}
            for each_image_path in images_paths:
                camera_key = each_image_path.split('/')[1]
                camera_dic[camera_key] = each_image_path
            if len(list(camera_dic.keys())) != 8 or len(list(camera_dic.values())) != 8:
                print('images_paths length not valid, short? ', camera_dic, images_paths, camera_dic, filename, map, split, frame_id)
                return None
            else:
                images_paths = list(camera_dic.values())
            assert len(images_paths) == 8, images_paths

        # check if image exists
        one_image_path = os.path.join(images_folder, images_paths[0])
        if not os.path.exists(one_image_path):
            print('image folder not exist: ', one_image_path)
            return None
        else:
            images = []
            for image_path in images_paths:
                image = PIL.Image.open(os.path.join(images_folder, image_path))
                image.thumbnail((1080 // 4, 1920 // 4))
                # image = image.resize((1080//4, 1920//4))
                # image = cv2.imread(os.path.join(images_folder, image_path))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image is None:
                    print('image is None: ', os.path.join(images_folder, image_path))
                images.append(np.array(image, dtype=np.float32))

            # shape: 8(cameras), 1080, 1920, 3
            result_to_return['camera_images'] = np.array(images, dtype=np.float32)
            del images

    if debug_raster_path is not None:
        # if debug_raster_path is not None:
        # check if path not exist, create
        if not os.path.exists(debug_raster_path):
            os.makedirs(debug_raster_path)
        image_file_name = sample['file_name'] + '_' + str(int(sample['frame_id']))
        # if split == 'test':
        if map == 'sg-one-north':
            save_result = save_raster(result_to_return, debug_raster_path, agent_types, len(sample_frames_in_past),
                                      image_file_name, split, high_res_scale, low_res_scale)
            if save_result and 'images_path' in sample:
                # copy camera images
                for camera in sample['images_path']:
                    import shutil
                    path_to_save = split + '_' + image_file_name + '_' + str(os.path.basename(camera))
                    shutil.copy(os.path.join(images_folder, camera), os.path.join(debug_raster_path, path_to_save))

    result_to_return["file_name"] = sample['file_name']
    result_to_return["map"] = sample['map']
    result_to_return["split"] = sample['split']
    result_to_return["frame_id"] = sample['frame_id']
    result_to_return["scenario_type"] = 'Unknown'
    if 'scenario_type' in sample:
        result_to_return["scenario_type"] = sample['scenario_type']
    if 'scenario_id' in sample:
        result_to_return["scenario_id"] = sample['scenario_id']
    if 't0_frame_id' in sample:
        result_to_return["t0_frame_id"] = sample['t0_frame_id']
    if 'intentions' in sample and kwargs.get('use_proposal', False):
        result_to_return["intentions"] = sample['intentions']

    result_to_return["route_ids"] = sample['route_ids']
    result_to_return["aug_current"] = aug_current
    # print('inspect shape: ', result_to_return['trajectory_label'].shape, result_to_return["context_actions"].shape)
    if 'off_roadx100' in kwargs.get('trajectory_prediction_mode', ''):
        """
        Pack road blocks as a list of all xyz points. Keep another list to mark the length of each block points.
        """
        route_blocks_pts = []
        route_block_ending_idx = []
        current_points_index = 0

        from shapely import geometry
        for each_route_id in sample['route_ids']:
            each_route_id = int(each_route_id)
            if each_route_id in sample['road_ids'] and each_route_id in data_dic['road_dic']:
                # point_num = len(data_dic['road_dic'][each_route_id]['xyz'])
                route_blocks_pts_this_block = data_dic['road_dic'][each_route_id]['xyz'][:, :2]
                # route_blocks_pts_this_block_line = geometry.LineString(route_blocks_pts_this_block).simplify(1)
                route_blocks_pts_this_block_line = geometry.LineString(route_blocks_pts_this_block)
                # turn line back to points
                route_blocks_pts_this_block = np.array(route_blocks_pts_this_block_line.coords.xy).transpose()
                route_blocks_pts_this_block = route_blocks_pts_this_block.flatten().tolist()
                point_num = len(route_blocks_pts_this_block) / 2
                route_blocks_pts += route_blocks_pts_this_block
                current_points_index += point_num * 2
                route_block_ending_idx.append(current_points_index)

        # for each_route_id in sample['route_ids']:
        #     each_route_id = int(each_route_id)
        #     if each_route_id in sample['road_ids'] and each_route_id in data_dic['road_dic']:
        #         point_num = len(data_dic['road_dic'][each_route_id]['xyz'])
        #         route_blocks_pts += data_dic['road_dic'][each_route_id]['xyz'][:, :2].flatten().tolist()
        #         current_points_index += point_num * 2
        #         route_block_ending_idx.append(current_points_index)

        # padding to the longest
        max_len = 1000 * 100
        route_blocks_pts = np.array(route_blocks_pts, dtype=np.float64)  # shape: block_num*2*point_num
        route_blocks_pts = np.pad(route_blocks_pts, (0, max_len - len(route_blocks_pts)))
        route_block_ending_idx = np.array(route_block_ending_idx, dtype=np.int32)
        route_block_ending_idx = np.pad(route_block_ending_idx, (0, 100 - len(route_block_ending_idx)))

        result_to_return['route_blocks_pts'] = route_blocks_pts
        result_to_return['route_block_ending_idx'] = route_block_ending_idx

    # if kwargs.get('pass_agents_to_model', False):
    #     pass
    result_to_return["ego_pose"] = origin_ego_pose

    # del agent_dic
    # del road_dic
    del ego_pose_agent_dic
    del data_dic

    return result_to_return

def transformation(ego_pose, context_action, label_trajectory):

    absolute_context_xy = context_action[:,:2] + ego_pose[:, :2]
    absolute_label_xy = label_trajectory[:,:2] + ego_pose[:, :2]
    return absolute_context_xy, absolute_label_xy

def inverse_transformation(ego_pose, context_action, label_trajectory):
    relative_context_xy = context_action[:, :2] - ego_pose[:, :2]
    relative_label_xy = label_trajectory[:, :2] - ego_pose[:, :2]
    return relative_context_xy, relative_label_xy
    
def augment_and_smooth_trajectory(trajectory, current_index, augment_x=1.0, augment_y=1.0, smooth_factor=0.5):
    """
    Augments the trajectory point at current_index and applies smoothing to the trajectory.
    
    Parameters:
    - trajectory: numpy array of shape (n, 2) where n is the number of points, and columns represent x and y coordinates.
    - current_index: index of the point to augment.
    - augment_x: maximum displacement along the x-axis as a fraction of the original x value.
    - augment_y: maximum displacement along the y-axis as a fraction of the original y value.
    - smooth_factor: controls how gradually changes are applied to maintain smoothness in transitions.
    
    Returns:
    - Updated trajectory with the augmented and smoothed point.
    """
    
    # Random shifts for x and y at the current index
    dx = (1 +np.random.random() /10 ) * augment_x
    dy = (1 +np.random.random() /10 ) * augment_y

    # Apply augmentations
    trajectory[current_index, 0] += dx
    trajectory[current_index, 1] += dy

    # Smoothing the trajectory
    # Starting from current_index to the start of the trajectory
    for i in range(current_index-1, 0, -1):
        trajectory[i, 0] = (trajectory[i, 0] * (1 - smooth_factor) + trajectory[i + 1, 0] * smooth_factor)
        trajectory[i, 1] = (trajectory[i, 1] * (1 - smooth_factor/2) + trajectory[i + 1, 1] * smooth_factor/2)
        
    # sin is used as the scheduler for the decay rate
    decay_rate = smooth_factor
    for i in range(current_index+1, current_index+20):
        decay_rate = np.sin(np.pi * (i - current_index - 10) / 20)/2 + 0.5
        trajectory[i, 0] = trajectory[current_index, 0] * (1 - decay_rate) + trajectory[i, 0] * decay_rate

    return trajectory

def nuplan_rasterize_collate_func_raw(batch, dic_path=None, autoregressive=False, save_index = False, **encode_kwargs):
    """
    'nuplan_collate_fn' is designed for nuplan dataset online generation.
    To use it, you need to provide a dictionary path of road dictionaries and agent&traffic dictionaries,  
    as well as a dictionary of rasterization parameters.

    The batch is the raw indexes data for one nuplan data item, each data in batch includes:
    road_ids, route_ids, traffic_ids, agent_ids, file_name, frame_id, map and timestamp.
    """
    # padding for tensor data
    expected_padding_keys = ["road_ids", "route_ids", "traffic_ids","navigation"]
    # expected_padding_keys = ["route_ids", "traffic_ids"]
    agent_id_lengths = list()
    for i, d in enumerate(batch):
        agent_id_lengths.append(len(d["agent_ids"]))
    max_agent_id_length = max(agent_id_lengths)
    for i, d in enumerate(batch):
        agent_ids = d["agent_ids"]
        agent_ids.extend(["null"] * (max_agent_id_length - len(agent_ids)))
        batch[i]["agent_ids"] = agent_ids

    padded_tensors = dict()
    for key in expected_padding_keys:
        tensors = [data[key] for data in batch]
        padded_tensors[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=-1)
        for i, _ in enumerate(batch):
            batch[i][key] = padded_tensors[key][i]

    # online rasterize
    map_func = partial(static_coor_rasterize_raw, data_path=dic_path, save_index = save_index, **encode_kwargs)
    # with ThreadPoolExecutor(max_workers=len(batch)) as executor:
    #     new_batch = list(executor.map(map_func, batch))
    new_batch = list()
    for i, d in enumerate(batch):
        rst = map_func(d)
        if rst is None:
            continue
        new_batch.append(rst)

    if len(new_batch) == 0:
        return {}
    
    # process as data dictionary
    result = dict()

    for key in new_batch[0].keys():
        if key is None:
            continue
        list_of_dvalues = []
        for d in new_batch:
            if d[key] is not None:
                list_of_dvalues.append(d[key])
            elif key == "scenario_type":
                list_of_dvalues.append('Unknown')
            else:
                print('Error: None value', key, d[key])   # scenario_type might be none for older dataset
        
        if key == "index":  
            result[key] = list_of_dvalues
        else:  
            result[key] = default_collate(list_of_dvalues)

    return result


def static_coor_rasterize_raw(sample, data_path, raster_shape=(224, 224),
                          frame_rate=20, past_seconds=2, future_seconds=8,
                          high_res_scale=4, low_res_scale=0.77,
                          road_types=20, agent_types=8, traffic_types=4,
                          past_sample_interval=2, future_sample_interval=2,
                          debug_raster_path=None, all_maps_dic=None, agent_dic=None,
                          frequency_change_rate=2,
                          save_index = False,
                          **kwargs):
    """
    WARNING: frame_rate has been change to 10 as default to generate new dataset pickles, this is automatically processed by hard-coded logits
    :param sample: a dictionary containing the following keys:
        - file_name: the name of the file
        - map: the name of the map, ex: us-ma-boston
        - split: the split, train, val or test
        - road_ids: the ids of the road elements
        - agent_ids: the ids of the agents in string
        - traffic_ids: the ids of the traffic lights
        - traffic_status: the status of the traffic lights
        - route_ids: the ids of the routes
        - frame_id: the frame id of the current frame, this is the global index which is irrelevant to frame rate of agent_dic pickles (20Hz)
        - debug_raster_path: if a debug_path past, will save rasterized images to disk, warning: will slow down the process
    :param data_path: the root path to load pickle files
    starting_frame, ending_frame, sample_frame in 20Hz,
    """

    filename = sample["file_name"]
    map = sample["map"]
    split = sample["split"]

    if split == 'val14_1k':
        split = 'val'
    elif split == 'test_hard14_index':
        split = 'test'

    frame_id = sample["frame_id"]  # current frame of this sample
    data_dic = load_data(sample, data_path, all_maps_dic)
    agent_dic = data_dic["agent_dic"]
    y_inverse = data_dic["y_inverse"]

    assert agent_dic['ego']['starting_frame'] == 0, f'ego starting frame {agent_dic["ego"]["starting_frame"]} should be 0'

    # augment frame id
    augment_frame_id = kwargs.get('augment_index', 0)
    if augment_frame_id != 0 and 'train' in split:
        frame_id += random.randint(-augment_frame_id - 1, augment_frame_id)
        frame_id = max(frame_id, past_seconds * frame_rate)

    # if new version of data, using relative frame_id
    relative_frame_id = True if 'starting_frame' in agent_dic['ego'] else False

    if "train" in split and kwargs.get('augment_current_pose_rate', 0) > 0:
        # copy agent_dic before operating to it
        ego_pose_agent_dic = agent_dic['ego']['pose'].copy()
    else:
        ego_pose_agent_dic = agent_dic['ego']['pose']

    # calculate frames to sample
    scenario_start_frame = frame_id - past_seconds * frame_rate
    scenario_end_frame = frame_id + future_seconds * frame_rate
    # for example,
    if kwargs.get('selected_exponential_past', True):
        # 2s, 1s, 0.5s, 0s
        # sample_frames_in_past = [scenario_start_frame + 0, scenario_start_frame + 20, scenario_start_frame + 30]
        sample_frames_in_past = [scenario_start_frame + 0, scenario_start_frame + 20, scenario_start_frame + 30, frame_id]
    elif kwargs.get('current_frame_only', False):
        sample_frames_in_past = [frame_id]
    else:
        # [10, 11, ...., 10+(2+8)*20=210], past_interval=2, future_interval=2, current_frame=50
        # sample_frames_in_past = [10, 12, 14, ..., 48], number=(50-10)/2=20
        sample_frames_in_past = list(range(scenario_start_frame, frame_id, past_sample_interval))  # add current frame in the end
    # sample_frames_in_future = [52, 54, ..., 208, 210], number=(210-50)/2=80
    sample_frames_in_future = list(range(frame_id + future_sample_interval, scenario_end_frame + future_sample_interval, future_sample_interval))  # + one step to avoid the current frame

    sample_frames = sample_frames_in_past + sample_frames_in_future
    # sample_frames = list(range(scenario_start_frame, frame_id + 1, frame_sample_interval))

    # augment current position
    aug_current = 0
    aug_rate = kwargs.get('augment_current_pose_rate', 0)
    if "train" in split and aug_rate > 0 and random.random() < aug_rate:
        augment_current_ratio = kwargs.get('augment_current_pose_ratio', 0.3)
        augment_current_with_past_linear_changes = kwargs.get('augment_current_with_past_linear_changes', False)
        augment_current_with_future_linear_changes = kwargs.get('augment_current_with_future_linear_changes', False)
        speed_per_step = nuplan_utils.euclidean_distance(
            ego_pose_agent_dic[frame_id // frequency_change_rate, :2],
            ego_pose_agent_dic[frame_id // frequency_change_rate - 5, :2]) / 5.0
        aug_x = augment_current_ratio * speed_per_step
        aug_y = augment_current_ratio * speed_per_step
        yaw_noise_scale = 0.05  # 360 * 0.05 = 18 degree
        aug_yaw = (random.random() * 2 - 1) * yaw_noise_scale
        dx = (random.random() * 2 - 1) * aug_x
        dy = (random.random() * 2 - 1) * aug_y
        dyaw = (random.random() * 2 * np.pi - np.pi) * aug_yaw
        ego_pose_agent_dic[frame_id//frequency_change_rate, 0] += dx
        ego_pose_agent_dic[frame_id//frequency_change_rate, 1] += dy
        ego_pose_agent_dic[frame_id//frequency_change_rate, -1] += dyaw
        aug_current = 1
        if augment_current_with_future_linear_changes:
            # linearly project the past poses
            # generate a numpy array decaying from 1 to 0 with shape of 80, 4
            decay = np.ones((80, 4)) * np.linspace(1, 0, 80).reshape(-1, 1)
            decay[:, 0] *= dx
            decay[:, 1] *= dy
            decay[:, 2] *= 0
            decay[:, 3] *= dyaw
            ego_pose_agent_dic[frame_id // frequency_change_rate: frame_id // frequency_change_rate + 80, :] += decay

        if augment_current_with_past_linear_changes:
            # generate a numpy array raising from 0 to 1 with the shape of 20, 4
            raising = np.ones((20, 4)) * np.linspace(0, 1, 20).reshape(-1, 1)
            raising[:, 0] *= dx
            raising[:, 1] *= dy
            raising[:, 2] *= 0
            raising[:, 3] *= dyaw
            ego_pose_agent_dic[frame_id // frequency_change_rate - 21: frame_id // frequency_change_rate - 1, :] += raising

    # initialize rasters
    origin_ego_pose = ego_pose_agent_dic[frame_id//frequency_change_rate].copy()  # hard-coded resample rate 2
    if kwargs.get('skip_yaw_norm', False):
        origin_ego_pose[-1] = 0

    if "agent_ids" not in sample.keys():
        if 'agent_ids_index' in sample.keys():
            agent_ids = []
            all_agent_ids = list(agent_dic.keys())
            for each_agent_index in sample['agent_ids_index']:
                if each_agent_index == -1:
                    continue
                if each_agent_index > len(all_agent_ids):
                    print(f'Warning: agent_ids_index is larger than agent_dic {each_agent_index} {len(all_agent_ids)}')
                    continue
                agent_ids.append(all_agent_ids[each_agent_index])
            assert 'ego' in agent_ids, 'ego should be in agent_ids'
        else:
            assert False
        # print('Warning: agent_ids not in sample keys')
        # agent_ids = []
        # max_dis = 300
        # for each_agent in agent_dic:
        #     starting_frame = agent_dic[each_agent]['starting_frame']
        #     target_frame = frame_id - starting_frame
        #     if target_frame < 0 or frame_id >= agent_dic[each_agent]['ending_frame']:
        #         continue
        #     pose = agent_dic[each_agent]['pose'][target_frame//frequency_change_rate, :].copy()
        #     if pose[0] < 0 and pose[1] < 0:
        #         continue
        #     pose -= origin_ego_pose
        #     if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
        #         continue
        #     agent_ids.append(each_agent)
    else:
        agent_ids = sample["agent_ids"]  # list of strings

    # num_frame = torch.div(frame_id, frequency_change_rate, rounding_mode='floor')
    # origin_ego_pose = agent_dic["ego"]["pose"][num_frame].copy()  # hard-coded resample rate 2
    if np.isinf(origin_ego_pose[0]) or np.isinf(origin_ego_pose[1]):
        assert False, f"Error: ego pose is inf {origin_ego_pose}, not enough precision while generating dictionary"

    rasters_high_res, rasters_low_res = draw_rasters(
        data_dic, origin_ego_pose, agent_ids,
        road_types, traffic_types, agent_types,
        sample_frames_in_past, frequency_change_rate,
        autoregressive=False,
        raster_shape=raster_shape,
        high_res_scale=high_res_scale,
        low_res_scale=low_res_scale,
        **kwargs
    )

    # context action computation
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])
    context_actions = list()
    ego_poses = ego_pose_agent_dic - origin_ego_pose
    rotated_poses = np.array([ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                              ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_,
                              np.zeros(ego_poses.shape[0]), ego_poses[:, -1]]).transpose((1, 0))
    rotated_poses[:, 1] *= y_inverse
    speed = None
    if kwargs.get('use_speed', True):
        # speed, old data dic does not have speed key
        speed = agent_dic['ego']['speed']  # v, a, angular_v
        if speed.shape[0] == ego_poses.shape[0] * 2:
            speed = speed[::2, :]
        for i in sample_frames_in_past:
            selected_pose = rotated_poses[i // frequency_change_rate]  # hard-coded frequency change
            selected_pose[-1] = normalize_angle(selected_pose[-1])
            action = np.concatenate((selected_pose, speed[i // frequency_change_rate]))
            context_actions.append(action)
    else:
        for i in sample_frames_in_past:
            action = rotated_poses[i//frequency_change_rate]  # hard-coded frequency change
            action[-1] = normalize_angle(action[-1])
            context_actions.append(action)

    # future trajectory
    # check if samples in the future is beyond agent_dic['ego']['pose'] length
    if relative_frame_id:
        sample_frames_in_future = (np.array(sample_frames_in_future, dtype=int) - agent_dic['ego']['starting_frame']) // frequency_change_rate
    if sample_frames_in_future[-1] >= ego_pose_agent_dic.shape[0]:
        # print('sample index beyond length of agent_dic: ', sample_frames_in_future[-1], agent_dic['ego']['pose'].shape[0])
        return None

    result_to_return = dict()
    trajectory_label = ego_pose_agent_dic[sample_frames_in_future, :].copy()
    raw_trajectory_label = ego_pose_agent_dic[scenario_start_frame//frequency_change_rate:scenario_end_frame//frequency_change_rate, :].copy()
    raw_trajectory_label = np.concatenate([raw_trajectory_label, speed[scenario_start_frame//frequency_change_rate:scenario_end_frame//frequency_change_rate, :]], axis=1)

    # get a planning trajectory from a CBC constant velocity planner
    # if kwargs.get('use_cbc_planner', False):
    #     from transformer4planning.rule_based_planner.nuplan_base_planner import MultiPathPlanner
    #     planner = MultiPathPlanner(road_dic=road_dic)
    #     planning_result = planner.plan_marginal_trajectories(
    #         my_current_pose=origin_ego_pose,
    #         my_current_v_mph=agent_dic['ego']['speed'][frame_id//frequency_change_rate, 0],
    #         route_in_blocks=sample['route_ids'].numpy().tolist(),
    #     )
    #     _, marginal_trajectories, _ = planning_result
    #     result_to_return['cbc_planning'] = marginal_trajectories
    trajectory_label -= origin_ego_pose
    traj_x = trajectory_label[:, 0].copy()
    traj_y = trajectory_label[:, 1].copy()
    trajectory_label[:, 0] = traj_x * cos_ - traj_y * sin_
    trajectory_label[:, 1] = traj_x * sin_ + traj_y * cos_
    trajectory_label[:, 1] *= y_inverse
    

    result_to_return["high_res_raster"] = np.array(rasters_high_res, dtype=bool)
    result_to_return["low_res_raster"] = np.array(rasters_low_res, dtype=bool)
    result_to_return["context_actions"] = np.array(context_actions, dtype=np.float32)
    result_to_return['trajectory_label'] = trajectory_label.astype(np.float32)
    result_to_return["raw_trajectory_label"] = raw_trajectory_label.astype(np.float32)
    if save_index:
        result_to_return["index"] = sample

    del rasters_high_res
    del rasters_low_res
    del trajectory_label
    del raw_trajectory_label
    # print('inspect: ', result_to_return["context_actions"].shape)

    camera_image_encoder = kwargs.get('camera_image_encoder', None)
    if camera_image_encoder is not None and 'test' not in split:
        import PIL.Image
        # load images
        if 'train' in split:
            images_folder = kwargs.get('train_camera_image_folder', None)
        elif 'val' in split:
            images_folder = kwargs.get('val_camera_image_folder', None)
        else:
            raise ValueError('split not recognized: ', split)

        images_paths = sample['images_path']
        if images_folder is None or len(images_paths) == 0:
            print('images_folder or images_paths not valid', images_folder, images_paths, filename, map, split, frame_id)
            return None
        if len(images_paths) != 8:
            # clean duplicate cameras
            camera_dic = {}
            for each_image_path in images_paths:
                camera_key = each_image_path.split('/')[1]
                camera_dic[camera_key] = each_image_path
            if len(list(camera_dic.keys())) != 8 or len(list(camera_dic.values())) != 8:
                print('images_paths length not valid, short? ', camera_dic, images_paths, camera_dic, filename, map, split, frame_id)
                return None
            else:
                images_paths = list(camera_dic.values())
            assert len(images_paths) == 8, images_paths

        # check if image exists
        one_image_path = os.path.join(images_folder, images_paths[0])
        if not os.path.exists(one_image_path):
            print('image folder not exist: ', one_image_path)
            return None
        else:
            images = []
            for image_path in images_paths:
                image = PIL.Image.open(os.path.join(images_folder, image_path))
                image.thumbnail((1080 // 4, 1920 // 4))
                # image = image.resize((1080//4, 1920//4))
                # image = cv2.imread(os.path.join(images_folder, image_path))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image is None:
                    print('image is None: ', os.path.join(images_folder, image_path))
                images.append(np.array(image, dtype=np.float32))

            # shape: 8(cameras), 1080, 1920, 3
            result_to_return['camera_images'] = np.array(images, dtype=np.float32)
            del images

    if debug_raster_path is not None:
        # if debug_raster_path is not None:
        # check if path not exist, create
        if not os.path.exists(debug_raster_path):
            os.makedirs(debug_raster_path)
        image_file_name = sample['file_name'] + '_' + str(int(sample['frame_id']))
        # if split == 'test':
        if map == 'sg-one-north':
            save_result = save_raster(result_to_return, debug_raster_path, agent_types, len(sample_frames_in_past),
                                      image_file_name, split, high_res_scale, low_res_scale)
            if save_result and 'images_path' in sample:
                # copy camera images
                for camera in sample['images_path']:
                    import shutil
                    path_to_save = split + '_' + image_file_name + '_' + str(os.path.basename(camera))
                    shutil.copy(os.path.join(images_folder, camera), os.path.join(debug_raster_path, path_to_save))

    result_to_return["file_name"] = sample['file_name']
    result_to_return["map"] = sample['map']
    result_to_return["split"] = sample['split']
    result_to_return["frame_id"] = sample['frame_id']
    result_to_return["scenario_type"] = 'Unknown'
    if 'scenario_type' in sample:
        result_to_return["scenario_type"] = sample['scenario_type']
    if 'scenario_id' in sample:
        result_to_return["scenario_id"] = sample['scenario_id']
    if 't0_frame_id' in sample:
        result_to_return["t0_frame_id"] = sample['t0_frame_id']
    if 'intentions' in sample and kwargs.get('use_proposal', False):
        result_to_return["intentions"] = sample['intentions']

    result_to_return["route_ids"] = sample['route_ids']
    result_to_return["aug_current"] = aug_current
    # print('inspect shape: ', result_to_return['trajectory_label'].shape, result_to_return["context_actions"].shape)
    if 'off_roadx100' in kwargs.get('trajectory_prediction_mode', ''):
        """
        Pack road blocks as a list of all xyz points. Keep another list to mark the length of each block points.
        """
        route_blocks_pts = []
        route_block_ending_idx = []
        current_points_index = 0

        from shapely import geometry
        for each_route_id in sample['route_ids']:
            each_route_id = int(each_route_id)
            if each_route_id in sample['road_ids'] and each_route_id in data_dic['road_dic']:
                # point_num = len(data_dic['road_dic'][each_route_id]['xyz'])
                route_blocks_pts_this_block = data_dic['road_dic'][each_route_id]['xyz'][:, :2]
                # route_blocks_pts_this_block_line = geometry.LineString(route_blocks_pts_this_block).simplify(1)
                route_blocks_pts_this_block_line = geometry.LineString(route_blocks_pts_this_block)
                # turn line back to points
                route_blocks_pts_this_block = np.array(route_blocks_pts_this_block_line.coords.xy).transpose()
                route_blocks_pts_this_block = route_blocks_pts_this_block.flatten().tolist()
                point_num = len(route_blocks_pts_this_block) / 2
                route_blocks_pts += route_blocks_pts_this_block
                current_points_index += point_num * 2
                route_block_ending_idx.append(current_points_index)

        # for each_route_id in sample['route_ids']:
        #     each_route_id = int(each_route_id)
        #     if each_route_id in sample['road_ids'] and each_route_id in data_dic['road_dic']:
        #         point_num = len(data_dic['road_dic'][each_route_id]['xyz'])
        #         route_blocks_pts += data_dic['road_dic'][each_route_id]['xyz'][:, :2].flatten().tolist()
        #         current_points_index += point_num * 2
        #         route_block_ending_idx.append(current_points_index)

        # padding to the longest
        max_len = 1000 * 100
        route_blocks_pts = np.array(route_blocks_pts, dtype=np.float64)  # shape: block_num*2*point_num
        route_blocks_pts = np.pad(route_blocks_pts, (0, max_len - len(route_blocks_pts)))
        route_block_ending_idx = np.array(route_block_ending_idx, dtype=np.int32)
        route_block_ending_idx = np.pad(route_block_ending_idx, (0, 100 - len(route_block_ending_idx)))

        result_to_return['route_blocks_pts'] = route_blocks_pts
        result_to_return['route_block_ending_idx'] = route_block_ending_idx

    # if kwargs.get('pass_agents_to_model', False):
    #     pass
    result_to_return["ego_pose"] = origin_ego_pose

    # del agent_dic
    # del road_dic
    del ego_pose_agent_dic
    del data_dic

    return result_to_return



def nuplan_rasterize_collate_func_diffusion(batch, dic_path=None, autoregressive=False, save_index = False, **encode_kwargs):
    """
    'nuplan_collate_fn' is designed for nuplan dataset online generation.
    To use it, you need to provide a dictionary path of road dictionaries and agent&traffic dictionaries,  
    as well as a dictionary of rasterization parameters.

    The batch is the raw indexes data for one nuplan data item, each data in batch includes:
    road_ids, route_ids, traffic_ids, agent_ids, file_name, frame_id, map and timestamp.
    """
    # padding for tensor data
    expected_padding_keys = ["road_ids", "route_ids", "traffic_ids","navigation","traffic_status"]
    # expected_padding_keys = ["route_ids", "traffic_ids"]
    agent_id_lengths = list()
    for i, d in enumerate(batch):
        agent_id_lengths.append(len(d["agent_ids"]))
    max_agent_id_length = max(agent_id_lengths)
    for i, d in enumerate(batch):
        agent_ids = d["agent_ids"]
        agent_ids.extend(["null"] * (max_agent_id_length - len(agent_ids)))
        batch[i]["agent_ids"] = agent_ids

    padded_tensors = dict()
    for key in expected_padding_keys:
        tensors = [torch.tensor(data[key]) for data in batch]
        padded_tensors[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=-1)
        for i, _ in enumerate(batch):
            batch[i][key] = padded_tensors[key][i]

    # online rasterize
    map_func = partial(static_coor_rasterize_diffusion, data_path=dic_path, save_index = save_index, **encode_kwargs)
    # with ThreadPoolExecutor(max_workers=len(batch)) as executor:
    #     new_batch = list(executor.map(map_func, batch))
    new_batch = list()
    for i, d in enumerate(batch):
        rst = map_func(d)
        if rst is None:
            continue
        new_batch.append(rst)

    if len(new_batch) == 0:
        return {}
    
    # process as data dictionary
    result = dict()

    for key in new_batch[0].keys():
        if key is None:
            continue
        list_of_dvalues = []
        for d in new_batch:
            if d[key] is not None:
                list_of_dvalues.append(d[key])
            elif key == "scenario_type":
                list_of_dvalues.append('Unknown')
            else:
                print('Error: None value', key, d[key])   # scenario_type might be none for older dataset
        
        if key == "index":  
            result[key] = list_of_dvalues
        else:  
            result[key] = default_collate(list_of_dvalues)

    return result


def static_coor_rasterize_diffusion(sample, data_path, raster_shape=(224, 224),
                          frame_rate=20, past_seconds=2, future_seconds=8,
                          high_res_scale=4, low_res_scale=0.77,
                          road_types=20, agent_types=8, traffic_types=4,
                          past_sample_interval=2, future_sample_interval=2,
                          debug_raster_path=None, all_maps_dic=None, agent_dic=None,
                          frequency_change_rate=2,
                          save_index = False,
                          **kwargs):
    """
    WARNING: frame_rate has been change to 10 as default to generate new dataset pickles, this is automatically processed by hard-coded logits
    :param sample: a dictionary containing the following keys:
        - file_name: the name of the file
        - map: the name of the map, ex: us-ma-boston
        - split: the split, train, val or test
        - road_ids: the ids of the road elements
        - agent_ids: the ids of the agents in string
        - traffic_ids: the ids of the traffic lights
        - traffic_status: the status of the traffic lights
        - route_ids: the ids of the routes
        - frame_id: the frame id of the current frame, this is the global index which is irrelevant to frame rate of agent_dic pickles (20Hz)
        - debug_raster_path: if a debug_path past, will save rasterized images to disk, warning: will slow down the process
    :param data_path: the root path to load pickle files
    starting_frame, ending_frame, sample_frame in 20Hz,
    """

    filename = sample["file_name"]
    map = sample["map"]
    split = sample["split"]
    augmented_trajectory = sample["augmented_trajectory"]
    augmented_trajectory = np.array(augmented_trajectory)

    if split == 'val14_1k':
        split = 'val'
    elif split == 'test_hard14_index':
        split = 'test'

    frame_id = sample["frame_id"]  # current frame of this sample
    data_dic = load_data(sample, data_path, all_maps_dic)
    agent_dic = data_dic["agent_dic"]
    y_inverse = data_dic["y_inverse"]

    assert agent_dic['ego']['starting_frame'] == 0, f'ego starting frame {agent_dic["ego"]["starting_frame"]} should be 0'

    # augment frame id
    augment_frame_id = kwargs.get('augment_index', 0)
    if augment_frame_id != 0 and 'train' in split:
        frame_id += random.randint(-augment_frame_id - 1, augment_frame_id)
        frame_id = max(frame_id, past_seconds * frame_rate)

    # if new version of data, using relative frame_id
    relative_frame_id = True if 'starting_frame' in agent_dic['ego'] else False

    if "train" in split and kwargs.get('augment_current_pose_rate', 0) > 0:
        # copy agent_dic before operating to it
        ego_pose_agent_dic = agent_dic['ego']['pose'].copy()
    else:
        ego_pose_agent_dic = agent_dic['ego']['pose']
    

    # calculate frames to sample
    scenario_start_frame = frame_id - past_seconds * frame_rate
    scenario_end_frame = frame_id + future_seconds * frame_rate
    print("gap: ", ego_pose_agent_dic[scenario_start_frame//frequency_change_rate:scenario_end_frame//frequency_change_rate, :] - augmented_trajectory[:, :4])
    ego_pose_agent_dic[scenario_start_frame//frequency_change_rate:scenario_end_frame//frequency_change_rate, :] = augmented_trajectory[:, :4]
    
    # for example,
    if kwargs.get('selected_exponential_past', True):
        # 2s, 1s, 0.5s, 0s
        # sample_frames_in_past = [scenario_start_frame + 0, scenario_start_frame + 20, scenario_start_frame + 30]
        sample_frames_in_past = [scenario_start_frame + 0, scenario_start_frame + 20, scenario_start_frame + 30, frame_id]
    elif kwargs.get('current_frame_only', False):
        sample_frames_in_past = [frame_id]
    else:
        # [10, 11, ...., 10+(2+8)*20=210], past_interval=2, future_interval=2, current_frame=50
        # sample_frames_in_past = [10, 12, 14, ..., 48], number=(50-10)/2=20
        sample_frames_in_past = list(range(scenario_start_frame, frame_id, past_sample_interval))  # add current frame in the end
    # sample_frames_in_future = [52, 54, ..., 208, 210], number=(210-50)/2=80
    sample_frames_in_future = list(range(frame_id + future_sample_interval, scenario_end_frame + future_sample_interval, future_sample_interval))  # + one step to avoid the current frame

    sample_frames = sample_frames_in_past + sample_frames_in_future
    # sample_frames = list(range(scenario_start_frame, frame_id + 1, frame_sample_interval))

    # augment current position
    aug_current = 0
    aug_rate = kwargs.get('augment_current_pose_rate', 0)
    if "train" in split and aug_rate > 0 and random.random() < aug_rate:
        augment_current_ratio = kwargs.get('augment_current_pose_ratio', 0.3)
        augment_current_with_past_linear_changes = kwargs.get('augment_current_with_past_linear_changes', False)
        augment_current_with_future_linear_changes = kwargs.get('augment_current_with_future_linear_changes', False)
        speed_per_step = nuplan_utils.euclidean_distance(
            ego_pose_agent_dic[frame_id // frequency_change_rate, :2],
            ego_pose_agent_dic[frame_id // frequency_change_rate - 5, :2]) / 5.0
        aug_x = augment_current_ratio * speed_per_step
        aug_y = augment_current_ratio * speed_per_step
        yaw_noise_scale = 0.05  # 360 * 0.05 = 18 degree
        aug_yaw = (random.random() * 2 - 1) * yaw_noise_scale
        dx = (random.random() * 2 - 1) * aug_x
        dy = (random.random() * 2 - 1) * aug_y
        dyaw = (random.random() * 2 * np.pi - np.pi) * aug_yaw
        ego_pose_agent_dic[frame_id//frequency_change_rate, 0] += dx
        ego_pose_agent_dic[frame_id//frequency_change_rate, 1] += dy
        ego_pose_agent_dic[frame_id//frequency_change_rate, -1] += dyaw
        aug_current = 1
        if augment_current_with_future_linear_changes:
            # linearly project the past poses
            # generate a numpy array decaying from 1 to 0 with shape of 80, 4
            decay = np.ones((80, 4)) * np.linspace(1, 0, 80).reshape(-1, 1)
            decay[:, 0] *= dx
            decay[:, 1] *= dy
            decay[:, 2] *= 0
            decay[:, 3] *= dyaw
            ego_pose_agent_dic[frame_id // frequency_change_rate: frame_id // frequency_change_rate + 80, :] += decay

        if augment_current_with_past_linear_changes:
            # generate a numpy array raising from 0 to 1 with the shape of 20, 4
            raising = np.ones((20, 4)) * np.linspace(0, 1, 20).reshape(-1, 1)
            raising[:, 0] *= dx
            raising[:, 1] *= dy
            raising[:, 2] *= 0
            raising[:, 3] *= dyaw
            ego_pose_agent_dic[frame_id // frequency_change_rate - 21: frame_id // frequency_change_rate - 1, :] += raising

    # initialize rasters
    origin_ego_pose = ego_pose_agent_dic[frame_id//frequency_change_rate].copy()  # hard-coded resample rate 2
    if kwargs.get('skip_yaw_norm', False):
        origin_ego_pose[-1] = 0

    if "agent_ids" not in sample.keys():
        if 'agent_ids_index' in sample.keys():
            agent_ids = []
            all_agent_ids = list(agent_dic.keys())
            for each_agent_index in sample['agent_ids_index']:
                if each_agent_index == -1:
                    continue
                if each_agent_index > len(all_agent_ids):
                    print(f'Warning: agent_ids_index is larger than agent_dic {each_agent_index} {len(all_agent_ids)}')
                    continue
                agent_ids.append(all_agent_ids[each_agent_index])
            assert 'ego' in agent_ids, 'ego should be in agent_ids'
        else:
            assert False
        # print('Warning: agent_ids not in sample keys')
        # agent_ids = []
        # max_dis = 300
        # for each_agent in agent_dic:
        #     starting_frame = agent_dic[each_agent]['starting_frame']
        #     target_frame = frame_id - starting_frame
        #     if target_frame < 0 or frame_id >= agent_dic[each_agent]['ending_frame']:
        #         continue
        #     pose = agent_dic[each_agent]['pose'][target_frame//frequency_change_rate, :].copy()
        #     if pose[0] < 0 and pose[1] < 0:
        #         continue
        #     pose -= origin_ego_pose
        #     if abs(pose[0]) > max_dis or abs(pose[1]) > max_dis:
        #         continue
        #     agent_ids.append(each_agent)
    else:
        agent_ids = sample["agent_ids"]  # list of strings

    # num_frame = torch.div(frame_id, frequency_change_rate, rounding_mode='floor')
    # origin_ego_pose = agent_dic["ego"]["pose"][num_frame].copy()  # hard-coded resample rate 2
    if np.isinf(origin_ego_pose[0]) or np.isinf(origin_ego_pose[1]):
        assert False, f"Error: ego pose is inf {origin_ego_pose}, not enough precision while generating dictionary"

    rasters_high_res, rasters_low_res = draw_rasters(
        data_dic, origin_ego_pose, agent_ids,
        road_types, traffic_types, agent_types,
        sample_frames_in_past, frequency_change_rate,
        autoregressive=False,
        raster_shape=raster_shape,
        high_res_scale=high_res_scale,
        low_res_scale=low_res_scale,
        **kwargs
    )

    # context action computation
    cos_, sin_ = math.cos(-origin_ego_pose[3]), math.sin(-origin_ego_pose[3])
    context_actions = list()
    ego_poses = ego_pose_agent_dic - origin_ego_pose
    rotated_poses = np.array([ego_poses[:, 0] * cos_ - ego_poses[:, 1] * sin_,
                              ego_poses[:, 0] * sin_ + ego_poses[:, 1] * cos_,
                              np.zeros(ego_poses.shape[0]), ego_poses[:, -1]]).transpose((1, 0))
    rotated_poses[:, 1] *= y_inverse
    speed = None
    if kwargs.get('use_speed', True):
        # speed, old data dic does not have speed key
        speed = agent_dic['ego']['speed']  # v, a, angular_v
        speed[scenario_start_frame//frequency_change_rate:scenario_end_frame//frequency_change_rate, :] = augmented_trajectory[:, 4:]
        if speed.shape[0] == ego_poses.shape[0] * 2:
            speed = speed[::2, :]
        for i in sample_frames_in_past:
            selected_pose = rotated_poses[i // frequency_change_rate]  # hard-coded frequency change
            selected_pose[-1] = normalize_angle(selected_pose[-1])
            action = np.concatenate((selected_pose, speed[i // frequency_change_rate]))
            context_actions.append(action)
    else:
        for i in sample_frames_in_past:
            action = rotated_poses[i//frequency_change_rate]  # hard-coded frequency change
            action[-1] = normalize_angle(action[-1])
            context_actions.append(action)

    # future trajectory
    # check if samples in the future is beyond agent_dic['ego']['pose'] length
    if relative_frame_id:
        sample_frames_in_future = (np.array(sample_frames_in_future, dtype=int) - agent_dic['ego']['starting_frame']) // frequency_change_rate
    if sample_frames_in_future[-1] >= ego_pose_agent_dic.shape[0]:
        # print('sample index beyond length of agent_dic: ', sample_frames_in_future[-1], agent_dic['ego']['pose'].shape[0])
        return None

    result_to_return = dict()
    trajectory_label = ego_pose_agent_dic[sample_frames_in_future, :].copy()
    # get a planning trajectory from a CBC constant velocity planner
    # if kwargs.get('use_cbc_planner', False):
    #     from transformer4planning.rule_based_planner.nuplan_base_planner import MultiPathPlanner
    #     planner = MultiPathPlanner(road_dic=road_dic)
    #     planning_result = planner.plan_marginal_trajectories(
    #         my_current_pose=origin_ego_pose,
    #         my_current_v_mph=agent_dic['ego']['speed'][frame_id//frequency_change_rate, 0],
    #         route_in_blocks=sample['route_ids'].numpy().tolist(),
    #     )
    #     _, marginal_trajectories, _ = planning_result
    #     result_to_return['cbc_planning'] = marginal_trajectories
    trajectory_label -= origin_ego_pose
    traj_x = trajectory_label[:, 0].copy()
    traj_y = trajectory_label[:, 1].copy()
    trajectory_label[:, 0] = traj_x * cos_ - traj_y * sin_
    trajectory_label[:, 1] = traj_x * sin_ + traj_y * cos_
    trajectory_label[:, 1] *= y_inverse
    

    result_to_return["high_res_raster"] = np.array(rasters_high_res, dtype=bool)
    result_to_return["low_res_raster"] = np.array(rasters_low_res, dtype=bool)
    result_to_return["context_actions"] = np.array(context_actions, dtype=np.float32)
    result_to_return['trajectory_label'] = trajectory_label.astype(np.float32)

    del rasters_high_res
    del rasters_low_res
    del trajectory_label
    # print('inspect: ', result_to_return["context_actions"].shape)

    camera_image_encoder = kwargs.get('camera_image_encoder', None)
    if camera_image_encoder is not None and 'test' not in split:
        import PIL.Image
        # load images
        if 'train' in split:
            images_folder = kwargs.get('train_camera_image_folder', None)
        elif 'val' in split:
            images_folder = kwargs.get('val_camera_image_folder', None)
        else:
            raise ValueError('split not recognized: ', split)

        images_paths = sample['images_path']
        if images_folder is None or len(images_paths) == 0:
            print('images_folder or images_paths not valid', images_folder, images_paths, filename, map, split, frame_id)
            return None
        if len(images_paths) != 8:
            # clean duplicate cameras
            camera_dic = {}
            for each_image_path in images_paths:
                camera_key = each_image_path.split('/')[1]
                camera_dic[camera_key] = each_image_path
            if len(list(camera_dic.keys())) != 8 or len(list(camera_dic.values())) != 8:
                print('images_paths length not valid, short? ', camera_dic, images_paths, camera_dic, filename, map, split, frame_id)
                return None
            else:
                images_paths = list(camera_dic.values())
            assert len(images_paths) == 8, images_paths

        # check if image exists
        one_image_path = os.path.join(images_folder, images_paths[0])
        if not os.path.exists(one_image_path):
            print('image folder not exist: ', one_image_path)
            return None
        else:
            images = []
            for image_path in images_paths:
                image = PIL.Image.open(os.path.join(images_folder, image_path))
                image.thumbnail((1080 // 4, 1920 // 4))
                # image = image.resize((1080//4, 1920//4))
                # image = cv2.imread(os.path.join(images_folder, image_path))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image is None:
                    print('image is None: ', os.path.join(images_folder, image_path))
                images.append(np.array(image, dtype=np.float32))

            # shape: 8(cameras), 1080, 1920, 3
            result_to_return['camera_images'] = np.array(images, dtype=np.float32)
            del images

    if debug_raster_path is not None:
        # if debug_raster_path is not None:
        # check if path not exist, create
        if not os.path.exists(debug_raster_path):
            os.makedirs(debug_raster_path)
        image_file_name = sample['file_name'] + '_' + str(int(sample['frame_id']))
        # if split == 'test':
        if map == 'sg-one-north':
            save_result = save_raster(result_to_return, debug_raster_path, agent_types, len(sample_frames_in_past),
                                      image_file_name, split, high_res_scale, low_res_scale)
            if save_result and 'images_path' in sample:
                # copy camera images
                for camera in sample['images_path']:
                    import shutil
                    path_to_save = split + '_' + image_file_name + '_' + str(os.path.basename(camera))
                    shutil.copy(os.path.join(images_folder, camera), os.path.join(debug_raster_path, path_to_save))

    result_to_return["file_name"] = sample['file_name']
    result_to_return["map"] = sample['map']
    result_to_return["split"] = sample['split']
    result_to_return["frame_id"] = sample['frame_id']
    result_to_return["scenario_type"] = 'Unknown'
    if 'scenario_type' in sample:
        result_to_return["scenario_type"] = sample['scenario_type']
    if 'scenario_id' in sample:
        result_to_return["scenario_id"] = sample['scenario_id']
    if 't0_frame_id' in sample:
        result_to_return["t0_frame_id"] = sample['t0_frame_id']
    if 'intentions' in sample and kwargs.get('use_proposal', False):
        result_to_return["intentions"] = sample['intentions']

    result_to_return["route_ids"] = sample['route_ids']
    result_to_return["aug_current"] = aug_current
    # print('inspect shape: ', result_to_return['trajectory_label'].shape, result_to_return["context_actions"].shape)
    if 'off_roadx100' in kwargs.get('trajectory_prediction_mode', ''):
        """
        Pack road blocks as a list of all xyz points. Keep another list to mark the length of each block points.
        """
        route_blocks_pts = []
        route_block_ending_idx = []
        current_points_index = 0

        from shapely import geometry
        for each_route_id in sample['route_ids']:
            each_route_id = int(each_route_id)
            if each_route_id in sample['road_ids'] and each_route_id in data_dic['road_dic']:
                # point_num = len(data_dic['road_dic'][each_route_id]['xyz'])
                route_blocks_pts_this_block = data_dic['road_dic'][each_route_id]['xyz'][:, :2]
                # route_blocks_pts_this_block_line = geometry.LineString(route_blocks_pts_this_block).simplify(1)
                route_blocks_pts_this_block_line = geometry.LineString(route_blocks_pts_this_block)
                # turn line back to points
                route_blocks_pts_this_block = np.array(route_blocks_pts_this_block_line.coords.xy).transpose()
                route_blocks_pts_this_block = route_blocks_pts_this_block.flatten().tolist()
                point_num = len(route_blocks_pts_this_block) / 2
                route_blocks_pts += route_blocks_pts_this_block
                current_points_index += point_num * 2
                route_block_ending_idx.append(current_points_index)

        # for each_route_id in sample['route_ids']:
        #     each_route_id = int(each_route_id)
        #     if each_route_id in sample['road_ids'] and each_route_id in data_dic['road_dic']:
        #         point_num = len(data_dic['road_dic'][each_route_id]['xyz'])
        #         route_blocks_pts += data_dic['road_dic'][each_route_id]['xyz'][:, :2].flatten().tolist()
        #         current_points_index += point_num * 2
        #         route_block_ending_idx.append(current_points_index)

        # padding to the longest
        max_len = 1000 * 100
        route_blocks_pts = np.array(route_blocks_pts, dtype=np.float64)  # shape: block_num*2*point_num
        route_blocks_pts = np.pad(route_blocks_pts, (0, max_len - len(route_blocks_pts)))
        route_block_ending_idx = np.array(route_block_ending_idx, dtype=np.int32)
        route_block_ending_idx = np.pad(route_block_ending_idx, (0, 100 - len(route_block_ending_idx)))

        result_to_return['route_blocks_pts'] = route_blocks_pts
        result_to_return['route_block_ending_idx'] = route_block_ending_idx

    # if kwargs.get('pass_agents_to_model', False):
    #     pass
    result_to_return["ego_pose"] = origin_ego_pose

    # del agent_dic
    # del road_dic
    del ego_pose_agent_dic
    del data_dic

    return result_to_return
