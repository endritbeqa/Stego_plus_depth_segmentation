from depth_dataset import ContrastiveDepthDataset
from src.modules.stego_modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm




@hydra.main(config_path="../../configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_directory_path = cfg.results_dir
    result_dir = join(result_directory_path, "results/predictions/images/")
    os.makedirs(join(result_dir, "real_img"), exist_ok=True)
    os.makedirs(join(result_dir, "segmentation_target"), exist_ok=True)


    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

    depth_transform_res = cfg.res

    if cfg.resize_to_original:
        depth_transform_res = cfg.resize_res

    loader_crop = "center"

    test_dataset = ContrastiveDepthDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.experiment_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(cfg.res, False, loader_crop),
        target_transform=get_transform(cfg.res, True, loader_crop),
        depth_transform=get_depth_transform(depth_transform_res, loader_crop),
        cfg=model.cfg,
    )

    loader = DataLoader(test_dataset, cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, collate_fn=flexible_collate)

    count_naming = 0
    count = 0

    # TODO Try to patch the image into 320x320 and then feed it into the transformer
    for i, batch in enumerate(tqdm(loader)):
        if count <= -1:
            count += 1
            continue

        with (torch.no_grad()):
            label = batch["label"]
            real_img = batch["real_img"]

            real_img = real_img[0].squeeze().numpy().astype(np.uint8)
            label = model.label_cmap[label[0].squeeze()].astype(np.uint8)

            real_img = Image.fromarray(real_img)
            label = Image.fromarray(label)


            real_img.save(join(result_dir,"real_img",str(count_naming)+".png"))
            label.save(join(result_dir, "segmentation_target", str(count_naming) + ".png"))

            count_naming+=1

            print("breakpoint")


if __name__ == "__main__":
    prep_args()
    my_app()
