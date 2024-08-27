##########################################################
# define the specific loss function for diffusion models #
##########################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

def offroad_loss(traj_logits,
                label,
                traj_loss,
                **kwargs):
    # WIP: checking off-road during training
    y_inverse = -1 if kwargs.get('map', None) == 'sg-one-north' else 1
    ego_poses = kwargs.get('ego_pose', None)  # bsz, 4
    route_blocks_pts = kwargs.get('route_blocks_pts', None)  # bsz, 100*1000
    route_block_ending_idx = kwargs.get('route_block_ending_idx', None)  # bsz, 100
    if ego_poses is not None and route_blocks_pts is not None:
        ego_poses[:, 0] *= y_inverse
        route_blocks_pts[:, 0] *= y_inverse
        # check if the predicted trajectory is off-road
        from transformer4planning.utils import nuplan_utils
        from shapely import geometry
        off_road_mask = torch.ones_like(label[...,
                                        :2], dtype=torch.float32)  # bsz, 80, 2, 1=off-road, 0=on-road

        examine_step = 5  # for efficiency, check every n step, do not change, not tested
        max_examin_frames = 40
        for i in range(traj_logits.shape[0]):
            # last_off = True
            # for j in range(traj_logits.shape[1]):
            # just check the first 30 frames
            off_road_mask[i, max_examin_frames:, :] = 0
            for j in range(max_examin_frames):
                if j % examine_step == 2:
                    # last_off = True
                    global_pred_point_t = nuplan_utils.change_coordination(traj_logits[i, j,
                                                                            :2].float().detach().cpu().numpy(),
                                                                            ego_poses[
                                                                                i].cpu().numpy(),
                                                                            ego_to_global=True)
                    # check if on road
                    route_blocks_xyz = []
                    previous_index = 0
                    for k in route_block_ending_idx[i]:
                        if k == 0:
                            # end with padding 0
                            break
                        route_blocks_xyz.append(route_blocks_pts[i,
                                                previous_index:k].cpu().numpy().reshape(-1, 2))
                        previous_index = k

                    for each_block_xyz in route_blocks_xyz:
                        # clean all padding 0s in route_block_xyz with a size of 1000, 3
                        each_block_xyz = each_block_xyz[each_block_xyz[:, 0] != 0]
                        if each_block_xyz.shape[0] == 0:
                            continue
                        block_line = geometry.LineString(each_block_xyz[:, :2])
                        current_point = geometry.Point(global_pred_point_t)
                        block_polygon = geometry.Polygon(block_line)
                        if block_polygon.contains(current_point):
                            off_road_mask[i, j-2:j+2, :] = 0
                            # last_off = False
                            break
                # else:
                #     if not last_off:
                #         off_road_mask[i, j, :] = 0

                if j == 2 and off_road_mask[i, j, 0] == 1:
                    # point zero off-road indicating wrong route info, ignore
                    off_road_mask[i, :, :] = 0
                    break
        # torch.set_printoptions(profile="full")
        # print("off_road_mask", off_road_mask[0, ::10])
        traj_loss[:,:,:2] *= (1 + off_road_mask * 100)
        return traj_loss