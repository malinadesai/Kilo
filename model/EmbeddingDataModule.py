import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataprocessing import Embedding_Data

class EmbeddingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tensor_paths,
        detection_limits,
        data_fillers,
        batch_size=100,
        train_frac=0.8,
        val_frac=0.1
    ):
        super().__init__()
        self.tensor_paths = tensor_paths
        self.detection_limits = detection_limits
        self.data_fillers = data_fillers
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.val_frac = val_frac

    def setup(self, stage=None):
        # Load data
        ds = [torch.load(p) for p in self.tensor_paths["data_shifted"]]
        du = [torch.load(p) for p in self.tensor_paths["data_unshifted"]]
        ps = [torch.load(p) for p in self.tensor_paths["param_shifted"]]
        pu = [torch.load(p) for p in self.tensor_paths["param_unshifted"]]

        data_shifted = torch.stack(ds)
        data_unshifted = torch.stack(du)
        param_shifted = torch.stack(ps)
        param_unshifted = torch.stack(pu)

        self.dataset = Embedding_Data(
            data_shifted, param_shifted, data_unshifted, param_unshifted,
            self.detection_limits, self.data_fillers
        )

        # Split
        n = len(self.dataset)
        train_size = int(self.train_frac * n)
        val_size = int(self.val_frac * n)
        test_size = n - train_size - val_size
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False)
