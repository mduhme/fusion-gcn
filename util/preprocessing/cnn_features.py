import torchvision.models as models
import torch

# set model download location
# torch.hub.set_dir()

resnet18 = models.resnet18(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)


def _remove_last_layer(model: torch.nn.Module) -> torch.nn.Module:
    return torch.nn.Sequential(*(list(model.children())[:-1]))


# Remove last layer
resnet18 = _remove_last_layer(resnet18)
squeezenet = _remove_last_layer(squeezenet)
inception = _remove_last_layer(inception)
googlenet = _remove_last_layer(googlenet)

models = [
    resnet18,
    squeezenet,
    inception,
    googlenet
]

print(squeezenet)
