import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime
import traceback
from functools import wraps
import asyncio
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Initialize Rich
console = Console()
install_rich_traceback(show_locals=True)

class VyperError(Exception):
    """Base exception class for Vyper"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": datetime.now().isoformat()
        }

class ConfigurationError(VyperError):
    """Raised when there's a configuration error"""
    pass

class APIError(VyperError):
    """Raised when there's an API-related error"""
    pass

class ValidationError(VyperError):
    """Raised when validation fails"""
    pass

class ProcessingError(VyperError):
    """Raised when processing fails"""
    pass

class AuthenticationError(VyperError):
    """Raised when authentication fails"""
    pass

class ResourceError(VyperError):
    """Raised when there's a resource-related error"""
    pass

def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    enable_rich: bool = True
) -> None:
    """Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        enable_rich: Whether to enable Rich logging
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"vyper_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    if enable_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(file_formatter)
    
    # Root logger config
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log an error with context
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    error_details = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat(),
        **(context or {})
    }
    
    logger.error(
        f"Error occurred: {error}",
        extra={
            "error_details": error_details
        }
    )

def handle_errors(
    error_types: Tuple[Exception, ...] = (Exception,),
    default_return: Any = None,
    log_errors: bool = True
) -> Callable:
    """Decorator for handling errors
    
    Args:
        error_types: Tuple of exception types to catch
        default_return: Default value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Decorated function that handles specified errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    log_error(logger, e, {
                        "function": func.__name__,
                        "args": args,
                        "kwargs": kwargs
                    })
                if isinstance(e, VyperError):
                    return {"error": e.to_dict()}
                return default_return

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except error_types as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    log_error(logger, e, {
                        "function": func.__name__,
                        "args": args,
                        "kwargs": kwargs
                    })
                if isinstance(e, VyperError):
                    return {"error": e.to_dict()}
                return default_return

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class ErrorBoundary:
    """Context manager for error handling"""
    
    def __init__(
        self,
        logger: logging.Logger,
        error_message: str,
        error_types: Tuple[Exception, ...] = (Exception,),
        on_error: Optional[Callable] = None,
        suppress_errors: bool = True
    ):
        self.logger = logger
        self.error_message = error_message
        self.error_types = error_types
        self.on_error = on_error
        self.suppress_errors = suppress_errors

    async def __aenter__(self) -> 'ErrorBoundary':
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> bool:
        if exc_val and isinstance(exc_val, self.error_types):
            log_error(self.logger, exc_val, {
                "boundary_message": self.error_message
            })
            if self.on_error:
                await self.on_error(exc_val)
            return self.suppress_errors
        return False

    def __enter__(self) -> 'ErrorBoundary':
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> bool:
        if exc_val and isinstance(exc_val, self.error_types):
            log_error(self.logger, exc_val, {
                "boundary_message": self.error_message
            })
            if self.on_error:
                asyncio.create_task(self.on_error(exc_val))
            return self.suppress_errors
        return False

class ErrorCollector:
    """Collect and aggregate errors"""
    
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.has_critical = False

    def add_error(
        self,
        error: Exception,
        severity: str = "error",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an error to the collection"""
        error_info = {
            "type": error.__class__.__name__,
            "message": str(error),
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
            **(context or {})
        }
        
        self.errors.append(error_info)
        if severity == "critical":
            self.has_critical = True

    def get_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        return {
            "total_errors": len(self.errors),
            "has_critical": self.has_critical,
            "errors_by_type": self._group_by_type(),
            "latest_error": self.errors[-1] if self.errors else None,
            "error_count_by_severity": self._count_by_severity()
        }

    def _group_by_type(self) -> Dict[str, int]:
        """Group errors by type"""
        error_counts = {}
        for error in self.errors:
            error_type = error["type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts

    def _count_by_severity(self) -> Dict[str, int]:
        """Count errors by severity"""
        severity_counts = {}
        for error in self.errors:
            severity = error["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts

    def clear(self) -> None:
        """Clear all collected errors"""
        self.errors.clear()
        self.has_critical = False