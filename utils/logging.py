import logging

from tensorboardX import SummaryWriter


def init_module_logger(name: str) -> logging.Logger:
    pass


def init_tensorboard_logger(log_dir: str) -> SummaryWriter:
    return SummaryWriter(log_dir)
