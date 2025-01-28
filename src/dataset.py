import torch
from torch.utils.data import Dataset
import numpy as np
import zarr
import cv2
import json
import os

def extract_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    object_name = data.get("pickable_object_name", "Unknown")
    points = [
        (point["location"]["x"], point["location"]["y"], point["location"]["z"])
        for point in data.get("points", [])
    ]
    
    return object_name, points

def read_zarr(zarr_file):
    z = zarr.open(zarr_file, mode='r')
    return z

def read_detection_json(target_class_obj, detection_json, voxel_size):
    
    object_name, points = extract_data(detection_json)
    raduis = target_class_obj.get_normalized_radius(object_name, voxel_size)
    idx = target_class_obj.get_class_idx(object_name)
    color = target_class_obj.get_class_color(object_name)

    detections = [(point[0], point[1], point[2], raduis, idx, color) for point in points]

    return detections

def read_tomogram(tomogram_path, voxel_size):
    print(tomogram_path)
    z = read_zarr(tomogram_path)['/0']
    print(z)
    tomogram = np.array(z)
    print(tomogram.shape)
    return tomogram

def create_tomogram_mask(tomogram, detections, voxel_size):
    mask = np.zeros(tomogram.shape, dtype=np.uint8)
    print(mask.shape)
    for x, y, z, raduis, idx, color in detections:
        print(x, y, z, raduis, idx, color)
        x, y, z = int(x), int(y), int(z)
        raduis = int(raduis)
        mask[:,:,z] = cv2.circle(mask[:,:,z], (y, x), raduis, idx, -1)
    
    return mask

def create_tomogram_masks(tomogram_paths, detection_paths, voxel_size):
    target_class_obj = TargetClasses()
    tomogram_path = tomogram_paths[0]
    tomogram = read_tomogram(tomogram_path, voxel_size)
    masks = []
    for detection_path in detection_paths:
        detections = read_detection_json(target_class_obj, detection_path, voxel_size)
        mask = create_tomogram_mask(tomogram, detections, voxel_size)
        masks.append(mask)
    
    return tomogram, masks

def generate_experiment_paths(dataset_dir):
    exp_names = os.listdir(dataset_dir + "/static/ExperimentRuns")
    tomogram_paths = [os.path.join(dataset_dir, "static/ExperimentRuns", exp_name, "VoxelSpacing10.000/denoised.zarr") for exp_name in exp_names]

    detection_paths = [ [os.path.join(dataset_dir, "overlay/ExperimentRuns", exp_name, "Picks/apo-ferritin.json") for exp_name in exp_names]]
    
    return tomogram_paths, detection_paths

def generate_dataset(dataset_dir):
    tomogram_paths, detection_paths = generate_experiment_paths(dataset_dir)
    voxel_size = 1
    tomogram, masks = create_tomogram_masks(tomogram_paths, detection_paths[0], voxel_size)
    return tomogram, masks


class TargetClasses:
    def __init__(self):
        self.n_classes = 6
        self.classes = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']
        self.class2idx = {c: i+1 for i, c in enumerate(self.classes)}
        self.idx2class = {i-1: c for i, c in enumerate(self.classes)}
        self.radius = [10, 10, 10, 10, 10, 10]
        self.colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]
    
    def __len__(self):
        return self.n_classes
    
    def get_normalized_radius(self, class_name, voxel_size):
        idx = self.class2idx[class_name]
        return self.radius[idx] / voxel_size
    
    def get_class_color(self, class_name):
        idx = self.class2idx[class_name]
        return self.colors[idx]
    
    def get_class_idx(self, class_name):
        return self.class2idx[class_name]

class TomographyDataset(Dataset):
    def __init__(self, tomograms, detection_jsons):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def main():
    dataset_dir = "/media/storage/Kaggle/dataset/train"
    tomogram, masks = generate_dataset(dataset_dir)
    print(tomogram.shape, len(masks))

if __name__ == "__main__":
    main()