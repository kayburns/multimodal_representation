import os
import h5py
import numpy as np
import ipdb
import cv2
from tqdm import tqdm

from torch.utils.data import Dataset


class TactoManipulationDataset(Dataset):
    """Multimodal Manipulation dataset."""

    def __init__(
        self,
        filename_list,
        transform=None,
        # episode_length=50,
        training_type="selfsupervised",
        # n_time_steps=1,
        action_dim=4,
        pairing_tolerance=0.06
    ):
        """
        Args:
            hdf5_file (handle): h5py handle of the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = filename_list
        self.transform = transform
        # self.episode_length = episode_length
        self.training_type = training_type
        # self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim
        self.pairing_tolerance = pairing_tolerance

        self._config_checks()
        self._init_paired_filenames()

    def __len__(self):
        return len(self.dataset_path) #* (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):
        idx = 0 #TODO: remove after debugging
        filename = self.dataset_path[idx]

        file_number, _ = self._parse_filename(filename)

        unpaired_filename, unpaired_idx = self.paired_filenames[idx]

        # if dataset_index >= self.episode_length - self.n_time_steps - 1:
        #     dataset_index = np.random.randint(
        #         self.episode_length - self.n_time_steps - 1
        #     )

        sample = self._get_single(
            filename,
            idx,
            unpaired_filename,
            unpaired_idx,
        )
        return sample

    def _get_single(
        self, filename, idx, unpaired_filename, unpaired_idx
    ):

        obs = self._load_from_file(filename)
        unpaired_obs = self._load_from_file(unpaired_filename)

        if self.training_type == "selfsupervised":

            image = obs['image0']
            depth = obs['depth0']
            proprio = obs['proprio0']
            # force = dataset["ee_forces_continuous"][dataset_index]
            tacto0 = obs["tacto00"]
            tacto1 = obs["tacto01"]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            # TODO: optical flow
            # flow = np.array(dataset["optical_flow"][dataset_index])
            # flow_mask = np.expand_dims(
            #     np.where(
            #         flow.sum(axis=2) == 0,
            #         np.zeros_like(flow.sum(axis=2)),
            #         np.ones_like(flow.sum(axis=2)),
            #     ),
            #     2,
            # )

            unpaired_image = image
            unpaired_depth = depth
            unpaired_proprio = unpaired_obs["proprio0"]
            unpaired_tacto0 = unpaired_obs["tacto00"]
            unpaired_tacto1 = unpaired_obs["tacto01"]

            sample = {
                "image": image,
                "depth": depth,
                # "flow": flow,
                # "flow_mask": flow_mask,
                "action": obs["action"],
                "tacto0": tacto0,
                "tacto1": tacto1,
                "proprio": proprio,
                # TODO: how to compute yaw with current action rep?
                # "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                # TODO: how to compute contact?
                # "contact_next": np.array(
                #     [dataset["contact"][dataset_index + 1].sum() > 0]
                # ).astype(np.float),
                "unpaired_image": unpaired_image,
                "unpaired_tacto0": unpaired_tacto0,
                "unpaired_tacto1": unpaired_tacto1,
                "unpaired_proprio": unpaired_proprio,
                "unpaired_depth": unpaired_depth,
            }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_from_file(self, filename):
        obs = {}
        obs['proprio0'] = np.load(os.path.join(filename, 'proprio_0.npy'))
        obs['proprio1'] = np.load(os.path.join(filename, 'proprio_1.npy'))
        # image
        obs['image0'] = cv2.imread(os.path.join(filename, 'cam_color_0.png'))
        obs['image1'] = cv2.imread(os.path.join(filename, 'cam_color_1.png'))
        # depth
        obs['depth0'] = cv2.imread(os.path.join(filename, 'cam_depth_0.png'))
        obs['depth1'] = cv2.imread(os.path.join(filename, 'cam_depth_1.png'))
        # tacto (replaces force)
        # TODO: what is indexing? (obs num, joint num)?
        # TODO: add depth image
        obs['tacto00'] = cv2.imread(os.path.join(filename, 'digits_color_0_0.png'))
        obs['tacto01'] = cv2.imread(os.path.join(filename, 'digits_color_0_1.png'))
        obs['tacto10'] = cv2.imread(os.path.join(filename, 'digits_color_1_0.png'))
        obs['tacto11'] = cv2.imread(os.path.join(filename, 'digits_color_1_1.png'))
        # action
        obs['action'] = np.load(os.path.join(filename, 'action_vec.npy'))
        return obs

    def _init_paired_filenames(self):
        """
        Precalculates the paired filenames.
        Imposes a distance tolerance between paired images
        """
        tolerance = self.pairing_tolerance

        # all_combos = set()

        self.paired_filenames = {}
        for idx in tqdm(range(len(self.dataset_path)), desc="pairing_files"):
            filename = self.dataset_path[idx]
            file_number, _ = self._parse_filename(filename)

            # obs = self._load_from_file(filename)
            proprio = np.load(os.path.join(filename, 'proprio_0.npy'))

            proprio_dist = None
            while proprio_dist is None or proprio_dist < tolerance*2:
                # Get a random idx, file that is not the same as current
                unpaired_dataset_idx = np.random.randint(self.__len__())
                unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(unpaired_dataset_idx)

                while unpaired_filename == filename:
                    unpaired_dataset_idx = np.random.randint(self.__len__())
                    unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(unpaired_dataset_idx)

                # unpaired_obs = self._load_from_file(unpaired_filename)
                prorprio_unpaired = np.load(os.path.join(unpaired_filename, 'proprio_0.npy'))
                # TODO: check that this distance makes sense
                proprio_dist = np.linalg.norm(proprio - prorprio_unpaired)

            self.paired_filenames[idx] = (unpaired_filename, unpaired_idx)
            return
            # all_combos.add((unpaired_filename, unpaired_idx))

    def _idx_to_filename_idx(self, idx):
        """
        Utility function for finding info about a dataset index

        Args:
            idx (int): Dataset index

        Returns:
            filename (string): Filename associated with dataset index
            dataset_index (int): Index of data within that file
            list_index (int): Index of data in filename list
        """
        # list_index = idx // (self.episode_length - self.n_time_steps)
        # dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[idx]
        return filename, idx, idx

    def _parse_filename(self, filename):
        """ Parses the filename to get the file number and filename"""
        # if filename[-2] == "_":
        #     file_number = int(filename[-1])
        #     filename = filename[:-1]
        # else:
        #     file_number = int(filename[-2:])
        #     filename = filename[:-2]

        # return file_number, filename
        return int(filename.split("_")[-1]), filename

    def _config_checks(self):
        if self.training_type != "selfsupervised":
            raise ValueError(
                "Training type not supported: {}".format(self.training_type)
            )
