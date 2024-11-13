import logging
import time
from datetime import datetime
import os
from logging.handlers import RotatingFileHandler


class TimeStampedRotatingFileHandler(RotatingFileHandler):
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        # 生成时间戳
        now = datetime.now()
        timestamp = (
            now.strftime("%Y-%m-%d_%H-%M-%S") + f".{now.microsecond // 1000:03d}"
        )

        # 构造新文件名
        rollover_filename = f"{self.baseFilename}.{timestamp}"

        # 重命名当前日志文件
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, rollover_filename)

        # 删除多余的旧日志
        if self.backupCount > 0:
            # 找到所有历史日志文件
            log_dir = os.path.dirname(self.baseFilename)
            base = os.path.basename(self.baseFilename)
            files = [f for f in os.listdir(log_dir) if f.startswith(base + ".")]
            files = sorted(files, reverse=True)
            for old_file in files[self.backupCount :]:
                os.remove(os.path.join(log_dir, old_file))

        # 重新打开日志文件
        self.mode = "a"
        self.stream = self._open()


def configure_logger(
    prefix: str = "", backup_path: str = "", backup_size: int = 2 ** 30, backup_count: int = 7
):
    handlers = [logging.StreamHandler()]

    if backup_path:
        handler = TimeStampedRotatingFileHandler(
            backup_path, maxBytes=backup_size, backupCount=backup_count
        )
        handlers.append(handler)

    format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=handlers,
    )


logger = logging.getLogger(__name__)
configure_logger(backup_path=__file__ + ".log", backup_size=2 ** 20)

while True:
    logger.info(f"Log line")
