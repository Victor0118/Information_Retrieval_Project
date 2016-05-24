import time
import logging

logger = logging.getLogger(__name__)

def timed(fn):
    """Decorator used to benchmark functions runtime."""

    def wrapped(*arg, **kw):
        ts = time.time()
        result = fn(*arg, **kw)
        te = time.time()

        logger.info('Function = %s, Time = %2.6f sec' \
                    % (fn.__name__, (te - ts)))

        return result
    return wrapped