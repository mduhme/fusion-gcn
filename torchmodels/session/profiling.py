
from session.session import Session


class ProfilingSession(Session):
    def __init__(self, base_config):
        super().__init__(base_config, "profiling")

    def start(self, config: dict = None, **kwargs):
        # TODO implement validation only
        pass
