import os
import pandas as pd
import ray
from ray import tune

from progress import launch_tensorboard

from session.session import Session
from session.training import TrainingSession


class TuningSession(Session):
    def __init__(self, base_config):
        super().__init__(base_config, "tuning")

    def start(self, config: dict = None, **kwargs):
        super()._make_paths()
        results_dir = (os.environ.get("TEST_TMPDIR") or os.environ.get("TUNE_RESULT_DIR") or os.path.expanduser(
            "~/ray_results"))
        num_gpus = config.get("num_gpus", 1)
        num_cpus = config.get("num_cpus", 2)

        training_session = TrainingSession(self._base_config)
        training_session.disable_logging = True
        training_session.disable_checkpointing = True
        ray.init(include_dashboard=False, local_mode=True, num_gpus=num_gpus, num_cpus=num_cpus)

        tensorboard_url = launch_tensorboard(results_dir)
        print(f"TensorBoard launched: {tensorboard_url}.")

        analysis = tune.run(training_session.start, self.session_id, config=config, checkpoint_freq=20)
        result = analysis.get_best_trial("mean_accuracy")
        print("Best trial config: {}".format(result.config))
        print("Best trial final validation loss: {}".format(result.last_result["mean_loss"]))
        print("Best trial final validation accuracy: {}".format(result.last_result["mean_accuracy"]))

        df = analysis.dataframe("mean_accuracy", "max")

        df.to_pickle(os.path.join(self.log_path, "results.pkl"))
        with pd.ExcelWriter(os.path.join(self.log_path, "results.xlsx")) as writer:
            df.to_excel(writer)
