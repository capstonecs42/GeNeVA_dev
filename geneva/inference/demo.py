import os

from torch.utils.data import DataLoader
from cv2 import cv2

from geneva.data.datasets import DATASETS
from geneva.utils.config import keys, parse_config
from geneva.models.models import INFERENCE_MODELS
from geneva.data import codraw_dataset
from geneva.data import clevr_dataset


class Demonstration(object):

    def __init__(self, config, iteration=None):
        self.model = INFERENCE_MODELS[config.gan_type](config)

        dataset_path = config.demo_dataset
        model_path = config.load_snapshot
        self.model.load(model_path, iteration)
        self.dataset = DATASETS[config.dataset](path=keys[dataset_path],
                                                cfg=config,
                                                img_size=config.img_size)
        batch_size = len(self.dataset)
        if config.dataset == 'iclevr':
            batch_size = batch_size - 2
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
        self.iterations = 1

        if config.dataset == 'codraw':
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif config.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data

        if config.results_path is None:
            config.results_path = os.path.join(config.log_path, config.exp_name, 'results')
            if not os.path.exists(config.results_path):
                os.mkdir(config.results_path)

        self.config = config
        self.dataset_path = dataset_path

    def _get_encoded_prev_image(self):
        prev_image = self._get_prev_image_file()
        return cv2.imread(prev_image)

    def _get_prev_image_file(self):
        last_img_dir = sorted(os.listdir(self.config.results_path))[-2]
        files = os.path.join(self.config.results_path,last_img_dir)
        return os.listdir(files)[0]

    def demo(self):
        iter_data = iter(self.dataloader)
        batch = next(iter_data)
        if not self.config.use_prev_image:
            self.model.predict(batch)
        else:
            try:
                prev_image = self._get_encoded_prev_image()
                self.model.predict(batch, prev_image=prev_image)
            except IndexError:
                self.model.predict(batch)


if __name__ == '__main__':
    cfg = parse_config()
    demo = Demonstration(cfg)
    demo.demo()

