import os
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from progress import launch_tensorboard

from session.session import Session
from session.training import TrainingSession


class TuningSession(Session):
    def __init__(self, base_config, name: str = "tuning"):
        super().__init__(base_config, name)

    def start(self, config: dict = None, **kwargs):
        if config is None:
            raise ValueError("Argument 'config' must not be None")

        super()._make_paths()
        metric = kwargs.get("metric", "mean_accuracy")
        mode = kwargs.get("mode", "max")

        # better save results in home dir because paths may become very long
        # possibly exceeding the max path limit if stored in different paths
        results_dir = (os.environ.get("TEST_TMPDIR") or os.environ.get("TUNE_RESULT_DIR") or os.path.expanduser(
            "~/ray_results"))
        num_gpus = kwargs.get("num_gpus", 1)
        num_cpus = kwargs.get("num_cpus", 2)

        training_session = TrainingSession(self._base_config)
        training_session.disable_logging = True
        training_session.disable_checkpointing = True
        ray.init(include_dashboard=False, local_mode=True, num_gpus=num_gpus, num_cpus=num_cpus)

        tensorboard_url = launch_tensorboard(results_dir)
        print(f"TensorBoard launched: {tensorboard_url}.")

        scheduler = kwargs.get("scheduler", None)
        if scheduler is None:
            scheduler = ASHAScheduler(metric, mode=mode)
        analysis = tune.run(training_session.start, self.session_id, config=config, scheduler=scheduler,
                            checkpoint_freq=20, local_dir=results_dir)
        result = analysis.get_best_trial(metric)
        print("Best trial config: {}".format(result.config))
        print("Best trial final validation loss: {}".format(result.last_result["mean_loss"]))
        print("Best trial final validation accuracy: {}".format(result.last_result["mean_accuracy"]))

        # TODO search algorithm?
        df = analysis.dataframe(metric, mode)

        df.to_pickle(os.path.join(self.log_path, "results.pkl"))
        with pd.ExcelWriter(os.path.join(self.log_path, "results.xlsx")) as writer:
            df.to_excel(writer)
