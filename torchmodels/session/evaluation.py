
from session.session import Session


class EvaluationSession(Session):
    def __init__(self, base_config):
        super().__init__(base_config, "evaluation")

    def start(self, config: dict = None, **kwargs):
        # TODO implement validation only
        pass
