import numpy as np
import itertools as it
from collections import namedtuple
from operator import attrgetter

class ItemPool():

    '''
    represents items in namedtuple form
    '''

    def __init__(self, feature_dict):

        '''
        args
        ----
        feature_dict : dict 
            dict of [feature_name (key) -> n_classes (val)] pairs
            e.g., {'shape':6, 'color':6}
        '''

        self.feature_dict = feature_dict
        self.Item = namedtuple('Item', feature_dict.keys())
        self.all_items = self._generate_all_items()

    def _generate_all_items(self):
        keys = self.feature_dict.keys()
        items = list(it.product(*[np.arange(self.feature_dict[k]) for k in keys]))
        items = [self.Item(**{k:x[i] for i, k in enumerate(keys)}) for x in items]
        return items
    
    def sample(self, n=1, replace=False):
        if not replace:
            assert n <= len(self.all_items)
        item_indices = np.random.choice(range(len(self.all_items)),  size=n, replace=replace)
        items = [self.all_items[i] for i in item_indices]
        return items

class Solver():

    '''
    handles ground-truth task operations on a given sequence
    '''

    def __init__(self):

        self.solvers = {'copy': self.copy, 
                        'reverse': self.reverse, 
                        'sort_by_shape': self.sort_by_shape, 
                        'reverse_sort_by_shape': self.reverse_sort_by_shape,
                        'sort_by_shape_first': self.sort_by_shape_first,
                        'sort_by_color': self.sort_by_color,
                        'reverse_sort_by_color': self.reverse_sort_by_color,
                        'sort_by_color_first': self.sort_by_color_first,
                        'sort_by_texture': self.sort_by_texture,
                        'reverse_sort_by_texture': self.reverse_sort_by_texture,
                        'sort_by_texture_first': self.sort_by_texture_first}

    def solve(self, x, task):
        '''
        args
        ----
        x : list
            input sequence of items
        task_str : str
            task string identifier
        '''
        if task in self.solvers.keys():
            return self.solvers[task](x)
        else:
            raise ValueError('task {%s} unknown to this solver' % task)
    
    @staticmethod
    def copy(items):
        return items

    @staticmethod
    def reverse(items):
        return list(reversed(items))

    @staticmethod
    def sort_by_shape(items):
        return sorted(items, key=attrgetter('shape'))

    @staticmethod
    def reverse_sort_by_shape(items):
        return sorted(items, key=attrgetter('shape'), reverse=True)

    @staticmethod
    def sort_by_shape_first(items):
        # TODO: don't hardcode item features
        return sorted(items, key=attrgetter('shape', 'color', 'texture'))

    @staticmethod
    def sort_by_color(items):
        return sorted(items, key=attrgetter('color'))

    @staticmethod
    def reverse_sort_by_color(items):
        return sorted(items, key=attrgetter('color'), reverse=True)

    @staticmethod
    def sort_by_color_first(items):
        # TODO: don't hardcode item features
        return sorted(items, key=attrgetter('color', 'shape', 'texture'))

    @staticmethod
    def sort_by_texture(items):
        return sorted(items, key=attrgetter('texture'))

    @staticmethod
    def reverse_sort_by_texture(items):
        return sorted(items, key=attrgetter('texture'), reverse=True)

    @staticmethod
    def sort_by_texture_first(items):
        return sorted(items, key=attrgetter('texture', 'shape', 'color'))