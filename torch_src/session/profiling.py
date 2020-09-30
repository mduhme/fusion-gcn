import torch
from torch.autograd.profiler import profile

from config import fill_model_config
from util.dynamic_import import import_dataset_constants
from progress import wrap_color, AnsiColors

from session.session import Session


class ProfilingSession(Session):
    def __init__(self, base_config, name: str = "profiling"):
        super().__init__(base_config, name)
        self.disable_logging = True
        self.disable_checkpointing = True

    def start(self, config: dict = None, **kwargs):
        """
        Start profiling: https://pytorch.org/tutorials/recipes/recipes/profiler.html
        """
        config = fill_model_config(config, self._base_config)

        num_batches = self._base_config.profiling_batches
        batch_size = config.get("batch_size", self._base_config.batch_size)
        data_shape, num_classes = import_dataset_constants(self._base_config.dataset, ["data_shape", "num_classes"])
        model, loss_function, optimizer, lr_scheduler = self._build_model(config)

        features_shape = (num_batches, batch_size, *data_shape)
        label_shape = (num_batches, batch_size)
        features = torch.randn(features_shape).float().cuda()
        labels = torch.zeros(label_shape, dtype=torch.long).random_(0, num_classes - 1).cuda()

        print(wrap_color(f"Run profiling {num_classes} batches...", AnsiColors.RED), end="")
        model.train()
        with profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:
            for x, y_true in zip(features, labels):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_function(y_pred, y_true)
                loss.backward()
                optimizer.step()

        print(wrap_color(f"\rRun profiling {num_classes} batches... Done.", AnsiColors.RED))
        print(wrap_color(prof.key_averages().table(sort_by="cuda_time_total"), AnsiColors.RED))
