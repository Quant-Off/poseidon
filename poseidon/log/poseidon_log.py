"""
Poseidon 로깅 클래스입니다.
"""

import logging
import os

import colorlog
from dotenv import load_dotenv

load_dotenv()

LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME")
if not LOG_FILE_PATH:
    LOG_FILE_PATH = "logs"
if not LOG_FILE_NAME:
    LOG_FILE_NAME = "poseidon.log"


class PoseidonLogger:
    """Poseidon 로깅 클래스입니다."""

    def __init__(self, critical="✦ CRITICAL ",
                 error="✦ ERROR ",
                 warning="✦ WARNING ",
                 info="✦ INFO ",
                 debug="✦ DEBUG "):
        # 로깅 레벨 설정 - 이름 변경
        critical = "✦ CRITICAL " if critical else "CRITICAL"
        error = "✦ ERROR " if error else "ERROR"
        warning = "✦ WARNING " if warning else "WARNING"
        info = "✦ INFO " if info else "INFO"
        debug = "✦ DEBUG " if debug else "DEBUG"
        logging.addLevelName(logging.CRITICAL, critical)
        logging.addLevelName(logging.ERROR, error)
        logging.addLevelName(logging.WARNING, warning)
        logging.addLevelName(logging.INFO, info)
        logging.addLevelName(logging.DEBUG, debug)

        # 로깅 설정 - 색상 및 포맷 개선
        self.logger = logging.getLogger()
        # 환경변수로 로깅 레벨 제어 (기본값: INFO)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))

        # 파일 핸들러 (색상 없음)
        self.file_handler = logging.FileHandler(os.path.join(LOG_FILE_PATH, LOG_FILE_NAME), encoding="utf-8")
        self.file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-12s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.file_handler.setFormatter(self.file_formatter)

        # 콘솔 핸들러 (색상 적용)
        self.console_handler = colorlog.StreamHandler()
        self.console_formatter = colorlog.ColoredFormatter(
            "%(light_black)s%(asctime)s%(reset)s %(log_color)s%(levelname)-12s%(reset)s %(bold)s%(light_black)s│%(reset)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                debug: "black,bold,bg_cyan",
                info: "black,bold,bg_green",
                warning: "black,bold,bg_yellow",
                error: "black,bold,bg_red",
                critical: "red,bold,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        self.console_handler.setFormatter(self.console_formatter)

        # 핸들러 중복 추가 방지
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        has_console_handler = any(isinstance(h, colorlog.StreamHandler) for h in self.logger.handlers)
        
        if not has_file_handler:
            self.logger.addHandler(self.file_handler)
        if not has_console_handler:
            self.logger.addHandler(self.console_handler)

    def get_logger(self):
        return self.logger


__all__ = ['PoseidonLogger']
