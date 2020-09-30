from ray import tune

import session_helper


def make_tune_config(base_config) -> dict:
    config = {
        "base_lr": tune.grid_search([0.001, 0.005, 0.01, 0.05, 0.1]),
        "optimizer": tune.grid_search(list(session_helper.available_optimizers.keys())),
        "lr_scheduler": tune.grid_search(["None"] + list(session_helper.available_lr_schedulers.keys())),
        "lr_scheduler_args": {
            "milestones": (30, 40),
            "gamma": 0.1
        },
        "optimizer_args": {
            "nesterov": True,
            "momentum": 0.9,
            "weight_decay": tune.grid_search([0., 0.0001, 0.001, 0.01])
        },
        # "lr_scheduler": get_learning_rate_scheduler_search_config()
    }
    return config

# def get_optimizer_search_config():
#     return tune.sample_from(lambda spec: (
#         {
#             "SGD": {
#                 "momentum": 0.9,
#                 "nesterov": True,
#                 "weight_decay": tune.grid_search([0., 0.00001, 0.0001, 0.001, 0.01])
#             },
#             "ADAM": {
#
#             }
#         }[spec["config"]["optimizer"]]
#     ))


# def get_learning_rate_scheduler_search_config():
#     return tune.sample_from(lambda spec: (
#         {
#             "multistep": {
#                 "milestones": tune.grid_search([(30, 50), (50, 80)]),
#                 "gamma": tune.grid_search([0.01, 0.1, 0.5])
#             },
#             "None": {}
#         }[spec["config"]["lr_scheduler"]]
#     ))
#
#
# def get_tune_config() -> dict:
#     config = {
#         "base_lr": tune.grid_search([0.005, 0.01, 0.05, 0.1]),
#         "optimizer": "SGD",
#         "lr_scheduler": tune.grid_search(["None"] + list(available_lr_schedulers.keys())),
#         "lr_scheduler_args": {
#             "milestones": (30, 40),
#             "gamma": 0.1
#         },
#         "optimizer_args": {
#             "nesterov": True,
#             "momentum": 0.9,
#             "weight_decay": tune.grid_search([0., 0.0001, 0.001, 0.01])
#         },
#         # "lr_scheduler": get_learning_rate_scheduler_search_config()
#     }
#     return config
