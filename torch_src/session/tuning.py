import os
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import tune_config
from progress import launch_tensorboard

from session.training import TrainingSession


class TuningSession(TrainingSession):
    def __init__(self, base_config, name: str = "tuning"):
        super().__init__(base_config, name)
        self.disable_logging = True
        self.disable_checkpointing = True

    def _start(self, config: dict = None, reporter=None):
        config = tune_config.prepare_tune_config(config)
        super().start(config, reporter=reporter)

    def start(self, config: dict = None, **kwargs):
        if config is None:
            raise ValueError("Argument 'config' must not be None")

        # better save results in home dir because paths may become very long
        # possibly exceeding the max path limit if stored in different paths
        results_dir = (os.environ.get("TEST_TMPDIR") or os.environ.get("TUNE_RESULT_DIR") or os.path.expanduser(
            "~/ray_results"))

        tune_log_path = os.path.join(results_dir, self.session_id)
        os.makedirs(tune_log_path, exist_ok=True)
        os.makedirs(self.out_path, exist_ok=True)
        metric = kwargs.get("metric", "mean_accuracy")
        mode = kwargs.get("mode", "max")

        num_gpus = kwargs.get("num_gpus", 1)
        num_cpus = kwargs.get("num_cpus", 1)

        ray.init(include_dashboard=False, local_mode=True, num_gpus=num_gpus, num_cpus=num_cpus)

        tensorboard_url = launch_tensorboard(tune_log_path)
        print(f"TensorBoard launched: {tensorboard_url}.")

        scheduler = kwargs.get("scheduler", None)
        if scheduler is None:
            scheduler = ASHAScheduler(metric, mode=mode)
        analysis = tune.run(self._start, self.session_id, config=config, scheduler=scheduler,
                            checkpoint_freq=20, local_dir=results_dir)
        result = analysis.get_best_trial(metric)
        print("Best trial config: {}".format(result.config))
        print("Best trial final validation loss: {}".format(result.last_result["mean_loss"]))
        print("Best trial final validation accuracy: {}".format(result.last_result["mean_accuracy"]))

        # TODO search algorithm?
        df = analysis.dataframe(metric, mode)
        df.to_pickle(os.path.join(self.out_path, "results.pkl"))

        try:
            with pd.ExcelWriter(os.path.join(self.out_path, "results.xlsx")) as writer:
                df.to_excel(writer)
        except ModuleNotFoundError as e:
            print("Failed to write tuning result:", e)
