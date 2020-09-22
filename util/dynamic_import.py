
def import_class(name: str) -> type:
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_model(name: str, class_name: str = "Model") -> type:
    return import_class(f"{name}.{name}.{class_name}")
