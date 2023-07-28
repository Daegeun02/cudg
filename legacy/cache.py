from .gpuarray import gpuarray



class Cache:

    def __init__(self, dtype=gpuarray):
        self.cache_avl = {}
        self.cache_usg = {}
        self.dtype = dtype

    def request(self, shape):
        try:
            ary = self.cache_avl[shape].pop(0)
            # print("success to borrow cache")

        except KeyError:
            self.cache_avl[shape] = []
            self.cache_usg[shape] = []
            ary = self.dtype(shape, iscache=True)
            # print("fail to borrow cache, generate new one")

        except IndexError:
            ary = self.dtype(shape, iscache=True)
            # print("fail to borrow cache, generate new one")

        finally:
            self.cache_usg[shape] += [ary]
            return ary

    def return_cache(self, ary):
        self.cache_avl[ary.shape] += [ary]
        self.cache_usg[ary.shape].remove(ary)
        # print("success to return cache")

    def force_return(self):
        # print("forcing to take using cache")
        for key, value in self.cache_usg.items():
            ## take all usg cache
            self.cache_avl[key] += value
            ## clean usg cache
            self.cache_usg[key] = []

    def clean_up(self):
        for key, value in self.cache_avl.items():
            del(self.cache_avl[key])
