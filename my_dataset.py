import numpy as np
import torch
import torch.utils.data


class ECGs_Dataset(torch.utils.data.Dataset):
    def __init__(self, ecgs: str, diags: str):
        """Initializes Dataset with passed files.
        Args:
            ecgs: file of ecgs,
            diags: file of diagnoses.
        """
        self.ecgs = np.load(ecgs)
        self.diags = np.load(diags)

        # self.transforms = Compose(
        #     [
        #         ToTensor()
        #     ]
        # )

    def __getitem__(self, idx: int):
        """Returns the object by given index.
        Args:
            idx - index of the record.
        Returns:
            record and diagnosis.
        """

        record = self.ecgs[idx]
        diag = self.diags[idx]

        # return self.transforms(image), self.labels[idx]
        return record, diag

    def __len__(self):
        """Returns length of files containing in dataset."""

        return len(self.ecgs)
