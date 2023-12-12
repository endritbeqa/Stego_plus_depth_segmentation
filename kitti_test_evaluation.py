import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
from src.data.KITTI_dataset import *
from src.data.KITTI_data_utils import *
from train_segmentation import LitUnsupervisedSegmenter
from torch.utils.data import DataLoader
from src.utils.stego_utils import flexible_collate




@hydra.main( config_path="configs", config_name="kitti_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    ROOT_DIR=os.getcwd()
    result_root = cfg.output_path
    dataset_path = cfg.dataset_path

    output_directories = ["Predicted_Semantic", "Predicted_Instance", "Metrics", ]

    for directory in output_directories:
        os.makedirs(os.path.join(result_root, directory), exist_ok=True)

    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.model_path)
    print(model.cfg)

    image_transform = get_transform(cfg.res, is_target=False)
    target_transform = get_transform(cfg.res, is_target=True)
    point_cloud_to_depth_map = get_pointCloud_transform
    kitti_dataset = KITTI(root=os.path.join(ROOT_DIR,dataset_path),
                          image_transform=image_transform,
                          target_transform=target_transform,
                          point_cloud_transform=point_cloud_to_depth_map)

    calibration_matrixes = kitti_dataset.getCalibration_matrixes()

    data_loader = DataLoader(kitti_dataset, cfg.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             collate_fn=flexible_collate)

    model.eval().cuda()

    for i, batch in enumerate(tqdm(data_loader)):
        img = batch["img"]
        target_semantic = batch["semantic"]
        target_semantic_rgb = batch["semantic_rgb"]
        target_instance = batch["instance"]
        depth_map = batch["depth"]

        Image.fromarray(depth_map.squeeze().numpy().astype(np.uint8)).show()
        Image.fromarray(target_semantic_rgb.squeeze().numpy().astype(np.uint8)).show()
        print("Breakpoint")


if __name__ == "__main__":
    my_app()
