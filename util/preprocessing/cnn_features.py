import numpy as np
import torch
import torchvision.models as models

# set model download location
# torch.hub.set_dir()

resnet18 = models.resnet18(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
googlenet = models.googlenet(pretrained=True)


def _prepare_model(model: torch.nn.Module) -> torch.nn.Module:
    # Remove last layer
    layers_without_fc = list(model.children())[:-1]
    return torch.nn.Sequential(*layers_without_fc)


resnet18 = _prepare_model(resnet18)
squeezenet = _prepare_model(squeezenet)
googlenet = _prepare_model(googlenet)


models = {
    "resnet18": (resnet18, 512, lambda x: x, lambda x: x),
    "squeezenet": (squeezenet, 512, lambda x: x, lambda x: torch.mean(x, -1)),
    "googlenet": (googlenet, 1024, lambda x: x, lambda x: x)
}
default_model = "resnet18"


def get_feature_size(model_name: str = None) -> int:
    model_name = model_name or default_model
    return models[model_name][1]


def encode_sample(sample: np.ndarray, model_name: str = None, cuda=True) -> np.ndarray:
    model_name = model_name or default_model
    input_fn = models[model_name][2]
    output_fn = models[model_name][3]

    sample = input_fn(sample)

    if sample.ndim == 3:
        sample = np.expand_dims(sample, axis=0)

    sample = torch.from_numpy(sample)
    model = models[model_name][0]

    if cuda:
        sample = sample.cuda()
        model = model.cuda()

    model.eval()
    with torch.no_grad():
        feature = model(sample)
        feature = output_fn(feature)
        feature = torch.flatten(feature)

    if cuda:
        feature = feature.cpu()

    return feature.numpy()
