import os
import logging
import pickle

import pystan

logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.dirname(__file__))

__all__ = ['get_model']

def model_path(model_name):
    return os.path.join(ROOT, 'stan', model_name+'.stan')

def model_cache_path(model_name):
    return os.path.join(ROOT, 'stan', model_name+'.pkl')


def get_model(model_name, recompile=False):
    """ Get compiled StanModel
    This will compile the stan model if a cached pickle does not exist.

    model_name : str,
        model name without `.stan`
    recompile : bool
        If True, force recompilation even when cached model exists.
    """
    model_file = model_path(model_name)
    cache_file = model_cache_path(model_name)
    if (not os.path.exists(cache_file)) or recompile:
        model = pystan.StanModel(file=model_file)
        logger.info("Compiling {:s}".format(model_name))
        with open(cache_file, 'wb') as f:
            pickle.dump(model, f)
    else:
        logger.info('Reading model from disk')
        model = pickle.load(open(cache_file, 'rb'))
    return model
