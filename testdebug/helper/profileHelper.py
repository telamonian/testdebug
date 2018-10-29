import cProfile

__all__ = ['ProfileDecoratorFunc', 'ProfileDecorator', 'LineProfileDecorator']

def ProfileDecoratorFunc(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='tottime')
    return profiled_func

class ProfileDecorator(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = self.func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats(sort='tottime')
        return profiled_func

# modified from https://zapier.com/engineering/profiling-python-boss/
try:
    from line_profiler import LineProfiler

    def LineProfileDecorator(follow=None):
        if follow is None: follow = []

        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def LineProfileDecorator(follow=None):
        """Helpful if you accidentally leave in production!
        """
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner
