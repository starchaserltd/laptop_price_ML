import functools
import json
import traceback


def _error_as_json(ex, logger=None, status=500):
    if logger:
        logger.error(" -- Got exception in the tagger backend!")
        logger.error(" -- %r" % ex)
        logger.error(traceback.format_exc())
    return json.dumps({'error': "{}".format(ex)}), status


def wrap_exceptions(func, logger=None):
    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return _error_as_json(e, logger)
    return func_wrapper
