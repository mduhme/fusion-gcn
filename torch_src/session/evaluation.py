from session.session import Session


class EvaluationSession(Session):
    def __init__(self, base_config, name: str = "evaluation"):
        super().__init__(base_config, name)

    def start(self, config: dict = None, **kwargs):
        # TODO implement validation only
        print("EvaluationSession not yet implemented")
