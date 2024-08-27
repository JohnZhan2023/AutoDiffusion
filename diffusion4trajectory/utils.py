import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_trajectory(tensor):
    """
    Plot the trajectory based on a given tensor with shape 1x100x7, assuming the first two dimensions are x and y coordinates.
    
    Args:
        tensor (torch.Tensor): A tensor of shape 1x100x7, where the first two elements are assumed to be x and y coordinates.
    """
    # Ensure the input is a PyTorch tensor
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The input must be a PyTorch tensor.")
    
    # Check the shape of the tensor
    if tensor.shape != (1, 100, 7):
        raise ValueError("The shape of the tensor must be 1x100x7.")
    
    # Extract x and y coordinates
    x_coords = tensor[0, :, 0].numpy()  # Extract all x coordinates
    y_coords = tensor[0, :, 1].numpy()  # Extract all y coordinates
    
    # Plot the trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-')
    plt.title('Trajectory Plot')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()
