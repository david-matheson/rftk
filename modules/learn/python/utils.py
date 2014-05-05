def pop_kwargs(kwargs, argument_name, unused_kwargs_keys, defaultValue=None):
    if argument_name not in kwargs:
        if defaultValue is not None:
            return defaultValue
        else:
            raise Exception("%s is not in %s" % (argument_name, str(unused_kwargs_keys)))

    if argument_name in unused_kwargs_keys:
        unused_kwargs_keys.remove(argument_name)
    return kwargs.get(argument_name)