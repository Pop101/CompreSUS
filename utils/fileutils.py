import os


def set_extension(path, ext):
    # Remove current extension if any
    path = path.split(".")
    if len(path) > 1:
        path = path[:-1]
    path = ".".join(path)

    return f"{path}.{ext}"
