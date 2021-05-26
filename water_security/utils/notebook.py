import builtins


def isnotebook():
    return hasattr(builtins, "__IPYTHON__")
