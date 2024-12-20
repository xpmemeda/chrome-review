import contextlib
import logging
import os
import sys

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_all_loggers = []
_default_level_name = os.getenv("CR_LOGGING_LEVEL", "INFO")
_default_level = logging.getLevelName(_default_level_name.upper())


def set_log_file(fout, mode="a"):
    r"""Sets log output file.

    Args:
        fout: file-like object that supports write and flush, or string for the filename
        mode: specify the mode to open log file if *fout* is a string
    """
    if isinstance(fout, str):
        fout = open(fout, mode)
    LogFormatter.log_fout = fout


class LogFormatter(logging.Formatter):
    log_fout = None
    date_full = "[%(asctime)s %(name)s@%(filename)s:%(lineno)d] "
    date = "%(asctime)s "
    msg = "%(message)s"
    max_lines = 256

    def _color_exc(self, msg):
        r"""Sets the color of message as the execution type."""
        return "\x1b[34m{}\x1b[0m".format(msg)

    def _color_dbg(self, msg):
        r"""Sets the color of message as the debugging type."""
        return "\x1b[36m{}\x1b[0m".format(msg)

    def _color_warn(self, msg):
        r"""Sets the color of message as the warning type."""
        return "\x1b[1;31m{}\x1b[0m".format(msg)

    def _color_err(self, msg):
        r"""Sets the color of message as the error type."""
        return "\x1b[1;4;31m{}\x1b[0m".format(msg)

    def _color_omitted(self, msg):
        r"""Sets the color of message as the omitted type."""
        return "\x1b[35m{}\x1b[0m".format(msg)

    def _color_normal(self, msg):
        r"""Sets the color of message as the normal type."""
        return msg

    def _color_date(self, msg):
        r"""Sets the color of message the same as date."""
        return "\x1b[32m{}\x1b[0m".format(msg)

    def format(self, record: logging.LogRecord):
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, "DBG"
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, "WRN"
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, "ERR"
        else:
            mcl, mtxt = self._color_normal, ""

        if mtxt:
            mtxt += " "

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
            formatted = super(LogFormatter, self).format(record)
            nr_line = formatted.count("\n") + 1
            if nr_line >= self.max_lines:
                head, body = formatted.split("\n", 1)
                formatted = "\n".join(
                    [
                        head,
                        "BEGIN_LONG_LOG_{}_LINES{{".format(nr_line - 1),
                        body,
                        "}}END_LONG_LOG_{}_LINES".format(nr_line - 1),
                    ]
                )
            self.log_fout.write(formatted)
            self.log_fout.write("\n")
            self.log_fout.flush()

        self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super(LogFormatter, self).format(record)

        if record.exc_text or record.exc_info:
            # handle exception format
            b = formatted.find("Traceback ")
            if b != -1:
                s = formatted[b:]
                s = self._color_exc("  " + s.replace("\n", "\n  "))
                formatted = formatted[:b] + s

        nr_line = formatted.count("\n") + 1
        if nr_line >= self.max_lines:
            lines = formatted.split("\n")
            remain = self.max_lines // 2
            removed = len(lines) - remain * 2
            if removed > 0:
                mid_msg = self._color_omitted(
                    "[{} log lines omitted (would be written to output file "
                    "if set_log_file() has been called;\n"
                    " the threshold can be set at "
                    "LogFormatter.max_lines)]".format(removed)
                )
                formatted = "\n".join(lines[:remain] + [mid_msg] + lines[-remain:])

        return formatted

    def __set_fmt(self, fmt):
        self._style._fmt = fmt


def get_logger(name=None, formatter=LogFormatter):
    logger = logging.getLogger(name)
    if getattr(logger, "_init_done__", None):
        return logger
    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(_default_level)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter(datefmt=_DATE_FORMAT))
    handler.setLevel(0)
    del logger.handlers[:]
    logger.addHandler(handler)
    _all_loggers.append(logger)
    return logger


def set_log_level(level, update_existing=True):
    r"""Sets default logging level.

    Args:
        level: loggin level given by python :mod:`logging` module
        update_existing: whether to update existing loggers
    """
    global _default_level  # pylint: disable=global-statement
    origin_level = _default_level
    _default_level = level
    if update_existing:
        for i in _all_loggers:
            i.setLevel(level)
    return origin_level


_logger = get_logger(__name__)


def set_debug_log():
    r"""Sets logging level to debug for all components."""
    set_log_level(logging.DEBUG)


def set_info_log():
    r"""Sets logging level to info for all components."""
    set_log_level(logging.INFO)


def set_warning_log():
    r"""Sets logging level to warning for all components."""
    set_log_level(logging.WARNING)


def set_error_log():
    r"""Sets logging level to error for all components."""
    set_log_level(logging.ERROR)
