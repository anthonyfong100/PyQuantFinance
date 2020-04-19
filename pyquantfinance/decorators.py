def accepts(*types):
    def check_accepts(f):
        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                if not isinstance(a, t):
                    raise TypeError("arg %r does not match type %s" % (a, t))
            return f(*args, **kwds)
        new_f.__name__ = f.__name__
        return new_f
    return check_accepts
