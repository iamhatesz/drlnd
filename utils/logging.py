from datetime import datetime
import logging
import os

from tensorboardX import SummaryWriter


def get_run_id() -> str:
    if 'RUN_ID' not in os.environ:
        os.environ['RUN_ID'] = datetime.now().strftime('%Y%m%dT%H%M%S')
    return os.environ['RUN_ID']


def init_module_logger(name: str) -> logging.Logger:
    pass


def init_tensorboard_logger(log_dir: str, run_id: str) -> SummaryWriter:
    return SummaryWriter(os.path.join(log_dir, run_id))
