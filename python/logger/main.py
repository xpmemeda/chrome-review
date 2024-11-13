import logger

a_logger = logger.get_logger(__name__)
logger.set_debug_log()


a_logger.debug("hello, world")
a_logger.info("hello, world")
a_logger.warning("hello, world")
a_logger.error("hello, world")
