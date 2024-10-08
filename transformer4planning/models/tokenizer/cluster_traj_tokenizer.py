import torch
import pandas as pd
import numpy as np


class ClusterTrajTokenizer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # self.centers = torch.tensor(self.df[['label_center_x', 'label_center_y']].values, dtype=torch.float32)

        loaded_trajectories = []
        for _, row in self.df.iterrows():
            flat_traj = row[2:].to_numpy()
            # print("type", type(flat_traj), flat_traj.shape)
            traj = flat_traj.reshape((80, 3))
            loaded_trajectories.append(traj)
        data = np.array(loaded_trajectories)
        self.trajs = torch.tensor(data, dtype=torch.float32)

    def encode(self, key_points, dtype=None, device=None):
        """
        :param key_points: (N,2)
        :return: index of clusters (N,)
        """
        # key_points = torch.tensor(key_points, dtype=torch.float32)
        # self.centers = self.centers.type_as(key_points)

        # diff = key_points[:, None, :] - self.centers[None, :, :]
        # distances = torch.norm(diff, dim=2)
        # closest_cluster_indices = torch.argmin(distances, dim=1)
        # return closest_cluster_indices
        pass

    def decode(self, kp_ids, dtype=None, device=None):
        """
        :param kp_ids: index of clusters, (N,)
        :return: center points of clusters, (N, 2)
        """
        # kp_ids = torch.tensor(kp_ids, dtype=torch.int64)
        self.trajs = self.trajs.to(device=kp_ids.device)
        kp_trajs = self.trajs[kp_ids]
        return kp_trajs


if __name__ == "__main__":
    analyzer = ClusterTrajTokenizer("/lpai/output/models/cluster/kmeans_points_0_5s_1024.csv")
    # key_points = torch.tensor(np.array([[0.5, 1.2], [2.3, 3.4]]), dtype=torch.float32)
    # key_points_id = analyzer.encode(key_points)
    # print(key_points_id)
    # kp_ids = torch.tensor(np.array([0, 1]), dtype=torch.int64)
    # kp_centers = analyzer.decode(kp_ids)
    # print(kp_centers)
