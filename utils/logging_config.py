"""Logging configuration for the trading application."""

import logging
import sys
from io import TextIOWrapper
from typing import List


def _create_logging_handlers(log_file: str = 'trading_signals.log') -> List[logging.Handler]:
    """Create logging handlers that explicitly use UTF-8 encoding (helps on Windows consoles)."""
    handlers: List[logging.Handler] = []

    try:
        # File handler with UTF-8 encoding
        fh = logging.FileHandler(log_file, encoding='utf-8')
        handlers.append(fh)
    except Exception:
        # Fallback: File handler without explicit encoding
        try:
            handlers.append(logging.FileHandler(log_file))
        except Exception:
            pass

    # Safe console handler which avoids writing to a closed stream
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                stream = getattr(self, 'stream', None)

                # If stream is closed, missing, or not writable, fallback to sys.__stdout__
                if stream is None or getattr(stream, 'closed', False) or not hasattr(stream, 'write'):
                    fallback = getattr(sys, '__stdout__', None)
                    if fallback is None:
                        # Nothing to write to; skip emit
                        return
                    # Prefer wrapping fallback.buffer if available
                    fb = getattr(fallback, 'buffer', None)
                    if fb is not None:
                        try:
                            self.stream = TextIOWrapper(fb, encoding='utf-8', errors='replace')
                        except Exception:
                            # Use fallback as-is
                            self.stream = fallback
                    else:
                        self.stream = fallback

                # Ensure UTF-8 wrapper if underlying buffer exists and not already wrapped
                sbuf = getattr(self.stream, 'buffer', None)
                if sbuf is not None and not isinstance(self.stream, TextIOWrapper):
                    try:
                        self.stream = TextIOWrapper(sbuf, encoding='utf-8', errors='replace')
                    except Exception:
                        pass

                super().emit(record)
            except Exception:
                # Swallow to avoid crashing worker threads. Try to write a minimal error to stderr if possible.
                try:
                    err = getattr(sys, '__stderr__', None)
                    if err is not None and hasattr(err, 'write'):
                        eb = getattr(err, 'buffer', None)
                        msg = f"[Logging failure] {record.getMessage()}\n"
                        try:
                            if eb is not None:
                                es = TextIOWrapper(eb, encoding='utf-8', errors='replace')
                                es.write(msg)
                                try:
                                    es.flush()
                                except Exception:
                                    pass
                            else:
                                err.write(msg)
                                try:
                                    err.flush()
                                except Exception:
                                    pass
                        except Exception:
                            # Last resort: ignore
                            pass
                except Exception:
                    pass

    try:
        # prefer wrapping stdout but use SafeStreamHandler to handle closed streams
        sh = SafeStreamHandler(sys.stdout)
        handlers.append(sh)
    except Exception:
        handlers.append(SafeStreamHandler())

    return handlers


def setup_logging(log_file: str = 'logs/trading_signals.log'):
    """Configure enhanced logging for debugging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=_create_logging_handlers(log_file),
        force=True
    )
    
    # Set specific loggers to INFO/WARNING to reduce noise
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('requests').setLevel(logging.INFO)
    # Reduce verbosity from httpx/httpcore and OpenAI client (they emit lots of debug logs)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)
    # Suppress excessive yfinance DEBUG logging
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    # Suppress peewee database DEBUG logging
    logging.getLogger('peewee').setLevel(logging.WARNING)


# Initialize logger for this module
logger = logging.getLogger(__name__)
