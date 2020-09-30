import locale

locale.setlocale(locale.LC_ALL, "")

import torch_util
import session_helper
from config import get_configuration

if __name__ == "__main__":
    cf = get_configuration(tuple(session_helper.session_types.keys()),
                           tuple(session_helper.available_optimizers.keys()),
                           tuple(session_helper.available_lr_schedulers.keys()))
    if cf.fixed_seed is not None:
        torch_util.set_seed(cf.fixed_seed)

    session_type = session_helper.create_session(cf)
    session = session_type.instantiate(cf)
    session.start(session_type.create_config(cf))
