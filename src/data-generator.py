import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

class AnchorDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        image_files,
        mask_files,
        coord_files,
        img_dir,
        mask_dir,
        coord_source,
        batch_size,
        target_size,
        coord_vector_size,
        shuffle=True,
        is_train=True
    ):
        self.image_files = image_files
        self.mask_files = mask_files
        self.coord_files = coord_files
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.coord_source = coord_source
        self.batch_size = batch_size
        self.target_size = target_size
        self.coord_vector_size = coord_vector_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.on_epoch_end()

        assert len(self.image_files) == len(self.mask_files) == len(self.coord_files)

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment(self, img, mask):
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask

    def __getitem__(self, index):
        idxs = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.zeros((self.batch_size, *self.target_size, 3), dtype=np.float32)
        Y_mask = np.zeros((self.batch_size, *self.target_size, 1), dtype=np.float32)
        Y_anchor = np.zeros((self.batch_size, self.coord_vector_size), dtype=np.float32)

        for i, idx in enumerate(idxs):
            img = image.load_img(
                os.path.join(self.img_dir, self.image_files[idx]),
                target_size=self.target_size
            )
            mask = image.load_img(
                os.path.join(self.mask_dir, self.mask_files[idx]),
                color_mode="grayscale",
                target_size=self.target_size
            )

            img = img_to_array(img) / 255.0
            mask = img_to_array(mask) / 255.0

            coord_dicts = self.coord_source[self.coord_files[idx]]
            coords = []

            for d in coord_dicts:
                for p in d['points']:
                    coords.extend([
                        p[0] / self.target_size[1],
                        p[1] / self.target_size[0]
                    ])

            coords = coords[:self.coord_vector_size]
            coords += [0.0] * (self.coord_vector_size - len(coords))

            if self.is_train:
                img, mask = self._augment(img, mask)

            X[i] = img
            Y_mask[i] = mask
            Y_anchor[i] = coords

        return X, {
            "mask_output": Y_mask,
            "anchor_output": Y_anchor
        }