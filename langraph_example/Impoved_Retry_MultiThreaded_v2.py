#!/usr/bin/env python3
"""
Production-Grade Enhanced CronScheduler with Resilient URL Processing

This enterprise-ready script provides a comprehensive framework for downloading,
preprocessing, and ingesting files from URLs with advanced features including:
- External configuration management via a .ini file
- Exponential backoff retry logic for transient errors
- Token-bucket rate limiting to respect server policies
- Circuit breaker pattern to prevent cascading failures
- Live monitoring via HTTP health check and metrics endpoints (/health, /metrics)
- Advanced, rotating logs for application, errors, and metrics

--- USAGE ---
1. Install dependencies:
   pip install APScheduler requests

2. Create a configuration file named `url_processor_config.ini` in the same directory:
   (You can copy the example below)

3. Run the script:
   python your_script_name.py

4. Monitor the processor by accessing these URLs in your browser:
   - http://localhost:8080/health
   - http://localhost:8080/metrics
   - http://localhost:8080/status

--- EXAMPLE url_processor_config.ini ---
[processing]
pool_size = 5
timeout = 30
max_retries = 3
base_retry_delay = 1.0
chunk_size = 5
rate_limit_requests_per_minute = 60
circuit_breaker_failure_threshold = 5
circuit_breaker_timeout = 300

[paths]
download_dir = downloads
chunks_dir = chunks
log_dir = logs
results_dir = results

[logging]
level = INFO
format = %(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s

[monitoring]
health_check_port = 8080
metrics_enabled = true

[database]
vector_db_url = http://localhost:8000/api/ingest
"""

import os
import json
import time
import logging
import threading
import uuid
import shutil
import random
import signal
import sys
import math
import configparser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
from urllib.parse import urlparse
from collections import deque
from contextlib import contextmanager
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- Third-Party Imports ---
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR


# --- Configuration Management ---
class ConfigManager:
    """Manages external configuration from an .ini file."""
    def __init__(self, config_file: str = "url_processor_config.ini"):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()
        self._load_defaults()
        self._load_from_file()

    def _load_defaults(self):
        """Loads hardcoded default configuration values."""
        self.config.read_dict({
            'processing': {
                'pool_size': '10', 'timeout': '30', 'max_retries': '3',
                'base_retry_delay': '1.0', 'chunk_size': '5',
                'rate_limit_requests_per_minute': '60',
                'circuit_breaker_failure_threshold': '5', 'circuit_breaker_timeout': '300'
            },
            'paths': {'download_dir': 'downloads', 'chunks_dir': 'chunks', 'log_dir': 'logs', 'results_dir': 'results'},
            'logging': {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'},
            'monitoring': {'health_check_port': '8080', 'metrics_enabled': 'true'},
            'database': {'vector_db_url': 'http://localhost:8000/api/ingest'}
        })

    def _load_from_file(self):
        """Loads configuration from the .ini file, creating it if it doesn't exist."""
        if self.config_file.exists():
            self.config.read(self.config_file)
            logging.info(f"Configuration loaded from {self.config_file}")
        else:
            logging.warning(f"Configuration file not found. Creating default '{self.config_file}'.")
            with open(self.config_file, 'w') as f:
                self.config.write(f)

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Gets a configuration value, converting its type intelligently."""
        value = self.config.get(section, key, fallback=str(fallback) if fallback is not None else None)
        if value is None: return fallback
        if value.lower() in ('true', 'false'): return value.lower() == 'true'
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

# --- Advanced Logging ---
def setup_logging(config: ConfigManager):
    """Sets up advanced, rotating file-based logging."""
    from logging.handlers import RotatingFileHandler
    log_dir = Path(config.get('paths', 'log_dir'))
    log_dir.mkdir(exist_ok=True)
    formatter = logging.Formatter(config.get('logging', 'format'))
    log_level = getattr(logging, config.get('logging', 'level').upper(), logging.INFO)

    # Main handler
    main_handler = RotatingFileHandler(log_dir / 'processor.log', maxBytes=10MB, backupCount=5)
    main_handler.setFormatter(formatter)
    # Error-only handler
    error_handler = RotatingFileHandler(log_dir / 'processor_errors.log', maxBytes=10MB, backupCount=5)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=log_level, handlers=[main_handler, error_handler, console_handler])
    logging.info("--- Advanced logging initialized ---")

# --- Data Structures & Patterns ---
@dataclass
class ProcessingResult:
    """Enhanced data class for detailed processing results."""
    # This class was well-defined and is kept as is.
    url: str; status: str; message: str
    start_time: Optional[datetime] = None; end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None; thread_id: Optional[str] = None
    file_size_bytes: Optional[int] = None; chunks_created: Optional[int] = None
    ingestion_id: Optional[str] = None; retry_count: int = 0
    error_type: Optional[str] = None; rate_limited: bool = False
    circuit_breaker_triggered: bool = False

    def to_dict(self):
        d = asdict(self)
        d['start_time'] = self.start_time.isoformat() if self.start_time else None
        d['end_time'] = self.end_time.isoformat() if self.end_time else None
        return d

@dataclass
class ProcessingMetrics:
    """Comprehensive metrics tracking."""
    total_urls: int = 0; successful: int = 0; failed: int = 0
    retried: int = 0; rate_limited: int = 0; circuit_breaker_triggered: int = 0
    total_processing_time: float = 0.0; average_processing_time: float = 0.0

class RateLimiter:
    """Token bucket rate limiter."""
    def __init__(self, requests_per_minute: int):
        self.capacity = requests_per_minute
        self.tokens = threading.Semaphore(self.capacity)
        self.refill_thread = threading.Thread(target=self._refill, daemon=True)
        self.refill_thread.start()

    def _refill(self):
        """Refills tokens at a fixed rate."""
        while True:
            # Release one token at the required interval
            if self.tokens._value < self.capacity:
                 self.tokens.release()
            time.sleep(60.0 / self.capacity)

    def acquire(self) -> bool:
        """Acquires a token, blocking if necessary."""
        return self.tokens.acquire()

class CircuitBreaker:
    """A context-manager-based circuit breaker."""
    def __init__(self, failure_threshold: int, timeout: int):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = 'CLOSED'  # Can be CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
        self.lock = threading.Lock()

    @contextmanager
    def call(self):
        with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = 'HALF_OPEN'
                    self.failure_count = 0 # Allow one test call
                else:
                    raise ConnectionAbortedError("Circuit breaker is OPEN")
        try:
            yield
            with self.lock:
                if self.state == 'HALF_OPEN': self.state = 'CLOSED'
                self.failure_count = 0
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logging.error(f"Circuit breaker OPENED after {self.failure_count} failures.")
            raise e

# --- Health Check Server ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health and metrics endpoints."""
    # This class attribute will be set by the factory
    url_processor = None

    def do_GET(self):
        """Handles GET requests."""
        endpoints = {
            '/health': self._handle_health,
            '/metrics': self._handle_metrics,
            '/status': self._handle_status
        }
        handler_func = endpoints.get(self.path)
        if handler_func:
            handler_func()
        else:
            self.send_error(404, "Not Found")

    def _handle_health(self):
        is_healthy = self.url_processor is not None
        status_code = 200 if is_healthy else 503
        response = {'status': 'healthy' if is_healthy else 'unhealthy'}
        self._send_json(response, status_code)

    def _handle_metrics(self):
        metrics = asdict(self.url_processor.get_metrics())
        self._send_json(metrics)

    def _handle_status(self):
        status = {
            'active_threads': self.url_processor.get_active_threads(),
            'circuit_breaker_state': self.url_processor.circuit_breaker.state,
        }
        self._send_json(status)

    def _send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def log_message(self, format, *args):
        # Suppress noisy default logging
        pass

def HealthCheckServerFactory(processor):
    """Factory to correctly inject the processor into the handler."""
    HealthCheckHandler.url_processor = processor
    return HealthCheckHandler

class HealthCheckServer:
    """Manages the lifecycle of the health check HTTP server."""
    def __init__(self, processor, port: int):
        self.server = HTTPServer(('localhost', port), HealthCheckServerFactory(processor))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="HealthCheckServer")

    def start(self):
        self.thread.start()
        logging.info(f"Health check server started on http://localhost:{self.server.server_port}")

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        logging.info("Health check server stopped.")

# --- Enhanced URL Processor (Core Logic) ---
# ... This is the main class that orchestrates everything ...
# (The content of this class is extensive and provided in the final script below)

# --- Main Application (Scheduler & Execution) ---
# ... The final part of the script that brings everything together ...
# (Provided in the final script below)
# Final, complete script
#!/usr/bin/env python3
"""
Production-Grade Enhanced CronScheduler with Resilient URL Processing

This enterprise-ready script provides a comprehensive framework for downloading,
preprocessing, and ingesting files from URLs with advanced features including:
- External configuration management via a .ini file
- Exponential backoff retry logic for transient errors
- Token-bucket rate limiting to respect server policies
- Circuit breaker pattern to prevent cascading failures
- Live monitoring via HTTP health check and metrics endpoints (/health, /metrics)
- Advanced, rotating logs for application, errors, and metrics

--- USAGE ---
1. Install dependencies:
   pip install APScheduler requests

2. Create a configuration file named `url_processor_config.ini` in the same directory:
   (You can copy the example below)

3. Run the script:
   python your_script_name.py

4. Monitor the processor by accessing these URLs in your browser:
   - http://localhost:8080/health
   - http://localhost:8080/metrics
   - http://localhost:8080/status

--- EXAMPLE url_processor_config.ini ---
[processing]
pool_size = 5
timeout = 30
max_retries = 3
base_retry_delay = 1.0
chunk_size = 5
rate_limit_requests_per_minute = 60
circuit_breaker_failure_threshold = 5
circuit_breaker_timeout = 300

[paths]
download_dir = downloads
chunks_dir = chunks
log_dir = logs
results_dir = results

[logging]
level = INFO
format = %(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s

[monitoring]
health_check_port = 8080
metrics_enabled = true

[database]
vector_db_url = http://localhost:8000/api/ingest
"""

import os
import json
import time
import logging
import threading
import uuid
import shutil
import random
import signal
import sys
import math
import configparser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
from urllib.parse import urlparse
from collections import deque
from contextlib import contextmanager
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- Third-Party Imports ---
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# --- Configuration Management ---
class ConfigManager:
    """Manages external configuration from an .ini file."""
    def __init__(self, config_file: str = "url_processor_config.ini"):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()
        self._load_defaults()
        self._load_from_file()

    def _load_defaults(self):
        """Loads hardcoded default configuration values."""
        self.config.read_dict({
            'processing': {
                'pool_size': '10', 'timeout': '30', 'max_retries': '3',
                'base_retry_delay': '1.0', 'chunk_size': '5',
                'rate_limit_requests_per_minute': '60',
                'circuit_breaker_failure_threshold': '5', 'circuit_breaker_timeout': '300'
            },
            'paths': {'download_dir': 'downloads', 'chunks_dir': 'chunks', 'log_dir': 'logs', 'results_dir': 'results'},
            'logging': {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'},
            'monitoring': {'health_check_port': '8080', 'metrics_enabled': 'true'},
            'database': {'vector_db_url': 'http://localhost:8000/api/ingest'}
        })

    def _load_from_file(self):
        """Loads configuration from the .ini file, creating it if it doesn't exist."""
        if self.config_file.exists():
            self.config.read(self.config_file)
            # Use logging after it's configured in main
        else:
            with open(self.config_file, 'w') as f:
                self.config.write(f)

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Gets a configuration value, converting its type intelligently."""
        value = self.config.get(section, key, fallback=str(fallback) if fallback is not None else None)
        if value is None: return fallback
        if value.lower() in ('true', 'false'): return value.lower() == 'true'
        try: return int(value)
        except ValueError:
            try: return float(value)
            except ValueError: return value

# --- Advanced Logging ---
def setup_logging(config: ConfigManager):
    """Sets up advanced, rotating file-based logging."""
    from logging.handlers import RotatingFileHandler
    log_dir = Path(config.get('paths', 'log_dir'))
    log_dir.mkdir(exist_ok=True)
    formatter = logging.Formatter(config.get('logging', 'format'))
    log_level = getattr(logging, config.get('logging', 'level').upper(), logging.INFO)
    
    # Define 10MB
    ten_mb = 10 * 1024 * 1024

    main_handler = RotatingFileHandler(log_dir / 'processor.log', maxBytes=ten_mb, backupCount=5)
    main_handler.setFormatter(formatter)
    error_handler = RotatingFileHandler(log_dir / 'processor_errors.log', maxBytes=ten_mb, backupCount=5)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=log_level, handlers=[main_handler, error_handler, console_handler])
    logging.info("--- Advanced logging initialized ---")
    if not config.config_file.exists():
        logging.warning(f"Configuration file not found. Created default at '{config.config_file}'.")
    else:
        logging.info(f"Configuration loaded from {config.config_file}")


# --- Data Structures & Patterns ---
@dataclass
class ProcessingResult:
    """Enhanced data class for detailed processing results."""
    url: str; status: str; message: str
    start_time: Optional[datetime] = None; end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None; thread_id: Optional[str] = None
    file_size_bytes: Optional[int] = None; chunks_created: Optional[int] = None
    ingestion_id: Optional[str] = None; retry_count: int = 0
    error_type: Optional[str] = None; rate_limited: bool = False
    circuit_breaker_triggered: bool = False

    def to_dict(self):
        d = asdict(self)
        d['start_time'] = self.start_time.isoformat() if self.start_time else None
        d['end_time'] = self.end_time.isoformat() if self.end_time else None
        return d

@dataclass
class ProcessingMetrics:
    """Comprehensive metrics tracking."""
    total_urls: int = 0; successful: int = 0; failed: int = 0
    retried: int = 0; rate_limited: int = 0; circuit_breaker_triggered: int = 0
    total_processing_time: float = 0.0; average_processing_time: float = 0.0

class RateLimiter:
    """A token bucket implementation for rate limiting."""
    def __init__(self, requests_per_minute: int):
        self.capacity = float(requests_per_minute)
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        """Acquires a token. Returns True on success."""
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens += elapsed * (self.capacity / 60)
            self.tokens = min(self.capacity, self.tokens)
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

class CircuitBreaker:
    """A context-manager-based circuit breaker."""
    def __init__(self, failure_threshold: int, timeout: int):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = 'CLOSED'
        self.last_failure_time = None
        self.lock = threading.Lock()

    @contextmanager
    def call(self):
        with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = 'HALF_OPEN'
                    logging.warning("Circuit breaker is now HALF-OPEN. Allowing one test call.")
                    self.failure_count = 0
                else:
                    raise ConnectionAbortedError("Circuit breaker is OPEN.")
        try:
            yield
            with self.lock:
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    logging.info("Circuit breaker is now CLOSED after successful test call.")
                self.failure_count = 0
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.state == 'HALF_OPEN' or self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logging.error(f"Circuit breaker OPENED after {self.failure_count} failures.")
            raise e

# --- Health Check Server ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health and metrics endpoints."""
    url_processor = None
    def do_GET(self):
        endpoints = {'/health': self._handle_health, '/metrics': self._handle_metrics, '/status': self._handle_status}
        if self.path in endpoints: endpoints[self.path]()
        else: self.send_error(404, "Not Found")
    def _handle_health(self): self._send_json({'status': 'healthy' if self.url_processor else 'unhealthy'})
    def _handle_metrics(self): self._send_json(asdict(self.url_processor.get_metrics()))
    def _handle_status(self): self._send_json({'active_threads': self.url_processor.get_active_threads(), 'circuit_breaker_state': self.url_processor.circuit_breaker.state})
    def _send_json(self, data, status=200): self.send_response(status); self.send_header('Content-type', 'application/json'); self.end_headers(); self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
    def log_message(self, format, *args): pass

def HealthCheckServerFactory(processor):
    HealthCheckHandler.url_processor = processor
    return HealthCheckHandler

class HealthCheckServer:
    """Manages the lifecycle of the health check HTTP server."""
    def __init__(self, processor, port: int):
        self.server = HTTPServer(('localhost', port), HealthCheckServerFactory(processor))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="HealthCheckServer")
    def start(self): self.thread.start(); logging.info(f"Health check server started on http://localhost:{self.server.server_port}")
    def stop(self): self.server.shutdown(); self.server.server_close(); self.thread.join(2); logging.info("Health check server stopped.")

# --- Enhanced URL Processor (Core Logic) ---
class EnhancedURLProcessor:
    """Production-grade URL processor with retries, rate limiting, and circuit breaking."""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.pool_size = config.get('processing', 'pool_size')
        self.executor = ThreadPoolExecutor(max_workers=self.pool_size, thread_name_prefix="URLProcessor")
        self.rate_limiter = RateLimiter(config.get('processing', 'rate_limit_requests_per_minute'))
        self.circuit_breaker = CircuitBreaker(config.get('processing', 'circuit_breaker_failure_threshold'), config.get('processing', 'circuit_breaker_timeout'))
        
        self.results: Dict[str, ProcessingResult] = {}
        self.results_lock = threading.RLock()
        self.metrics = ProcessingMetrics()
        self.metrics_lock = threading.RLock()
        
        self.download_dir = Path(config.get('paths', 'download_dir'))
        self.chunks_dir = Path(config.get('paths', 'chunks_dir'))
        self.results_dir = Path(config.get('paths', 'results_dir'))
        for d in [self.download_dir, self.chunks_dir, self.results_dir]: d.mkdir(exist_ok=True)
        
        if config.get('monitoring', 'health_check_port'):
            self.health_server = HealthCheckServer(self, config.get('monitoring', 'health_check_port'))
            self.health_server.start()
        else:
            self.health_server = None

    def process_urls(self, urls: List[str]):
        """Submits URLs to the thread pool for processing."""
        if not urls: logging.warning("No URLs provided to process."); return
        with self.metrics_lock: self.metrics.total_urls += len(urls)
        
        future_to_url = {self.executor.submit(self._process_single_url_with_retry, url): url for url in urls}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try: future.result() # Logged within the retry logic
            except Exception as e: logging.critical(f"Future for {url} completed with unexpected error: {e}", exc_info=True)

    def _process_single_url_with_retry(self, url: str) -> ProcessingResult:
        """Processes a single URL with exponential backoff retry logic."""
        thread_id, start_time, retry_count, last_exception = threading.current_thread().name, datetime.now(), 0, None
        with self.results_lock: self.results[url] = ProcessingResult(url=url, status='processing', start_time=start_time, thread_id=thread_id)

        while retry_count <= self.config.get('processing', 'max_retries'):
            try:
                if not self.rate_limiter.acquire():
                    time.sleep(1) # Wait a moment if rate limited
                    raise ConnectionRefusedError("Rate limit exceeded")
                with self.circuit_breaker.call():
                    result = self._process_single_url(url, retry_count)
                    with self.metrics_lock: self.metrics.successful += 1
                    logging.info(f"✅ SUCCESS on attempt {retry_count+1} for {url}. Message: {result.message}")
                    return result
            except Exception as e:
                last_exception = e
                retry_count += 1
                if retry_count <= self.config.get('processing', 'max_retries'):
                    delay = min(self.config.get('processing', 'base_retry_delay') * (2**(retry_count-1)), 60) + random.uniform(0, 1)
                    logging.warning(f"Retrying ({retry_count}/{self.config.get('processing', 'max_retries')}) for {url} after {delay:.2f}s. Error: {e}")
                    time.sleep(delay)
                else: break # Max retries exceeded
        
        # Final failure after all retries
        error_type = type(last_exception).__name__
        result = ProcessingResult(
            url=url, status='failure', message=f"Failed after {retry_count-1} retries: {last_exception}",
            start_time=start_time, end_time=datetime.now(), thread_id=thread_id,
            retry_count=retry_count-1, error_type=error_type
        )
        with self.results_lock: self.results[url] = result
        with self.metrics_lock: self.metrics.failed += 1
        logging.error(f"❌ FAILURE for {url} after {result.retry_count} retries. Final error: {last_exception}")
        return result

    def _process_single_url(self, url: str, attempt: int) -> ProcessingResult:
        """The core processing pipeline for a single URL."""
        download_res = self._download_file(url)
        preprocess_res = self._preprocess_to_chunks(download_res['file_path'], url)
        ingest_res = self._ingest_to_vector_db(preprocess_res['chunks_dir'], url)
        
        result = self.results[url]
        result.status, result.message = 'success', f"Processed in {ingest_res['duration']:.2f}s"
        result.end_time, result.duration_seconds = datetime.now(), ingest_res['duration']
        result.file_size_bytes, result.chunks_created = download_res['file_size'], preprocess_res['chunks_count']
        result.ingestion_id, result.retry_count = ingest_res['ingestion_id'], attempt
        return result

    def _download_file(self, url: str) -> Dict[str, Any]:
        """Simulates downloading a file."""
        if "faildownload" in url or random.random() < 0.1: raise requests.RequestException("Simulated network failure")
        file_path = self.download_dir / f"{hashlib.md5(url.encode()).hexdigest()}.txt"
        file_path.write_text(f"Content from {url}", encoding='utf-8')
        return {'file_path': file_path, 'file_size': file_path.stat().st_size}

    def _preprocess_to_chunks(self, file_path: Path, url: str) -> Dict[str, Any]:
        """Simulates preprocessing a file into chunks."""
        if "failprocess" in url or random.random() < 0.1: raise RuntimeError("Simulated processing error")
        chunks_dir = self.chunks_dir / f"{file_path.stem}_chunks"
        chunks_dir.mkdir(exist_ok=True)
        return {'chunks_dir': chunks_dir, 'chunks_count': 5}

    def _ingest_to_vector_db(self, chunks_dir: Path, url: str) -> Dict[str, Any]:
        """Simulates ingesting chunks into a database."""
        if "failingest" in url or random.random() < 0.1: raise ConnectionError("Simulated DB connection error")
        return {'ingestion_id': str(uuid.uuid4()), 'duration': random.uniform(0.5, 2.0)}

    def get_metrics(self) -> ProcessingMetrics:
        """Returns the current processing metrics."""
        with self.metrics_lock:
            if self.metrics.successful > 0:
                self.metrics.average_processing_time = self.metrics.total_processing_time / self.metrics.successful
            return self.metrics

    def get_active_threads(self) -> int: return self.executor._work_queue.qsize() + len(self.executor._threads)
    def save_results(self):
        filename = self.results_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with self.results_lock: data = {url: res.to_dict() for url, res in self.results.items()}
        filename.write_text(json.dumps(data, indent=2), encoding='utf-8')
        logging.info(f"💾 Results for {len(data)} URLs saved to {filename}")

    def shutdown(self):
        """Gracefully shuts down the processor and its components."""
        logging.info("Shutting down URL processor...")
        if self.health_server: self.health_server.stop()
        self.executor.shutdown(wait=True)
        logging.info("URL processor shut down.")

# --- Main Application ---
class Application:
    """The main application class that orchestrates the scheduler and processor."""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.scheduler = BlockingScheduler()
        self.processor = EnhancedURLProcessor(config)
        self.url_lists = {"daily_batch": [
            "https://example.com/file1", "https://example.com/file2",
            "https://example.com/faildownload", "https://example.com/failprocess", "https://example.com/failingest",
            "https://httpstat.us/404", "https://httpstat.us/503"
        ]}
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run(self):
        """Starts the application."""
        logging.info("--- Starting Application ---")
        # For demonstration, run the job immediately instead of on a cron schedule.
        self.url_processing_job("daily_batch")
        # In production, you would use:
        # self.scheduler.add_job(self.url_processing_job, CronTrigger.from_crontab('0 8 * * *'), args=["daily_batch"])
        # self.scheduler.start()
        
        # Since we ran it once, we can now shut down.
        self.stop()

    def url_processing_job(self, list_name: str):
        """The main job function called by the scheduler."""
        logging.info(f"🚀🚀🚀 Starting URL Processing Job: {list_name} 🚀🚀🚀")
        urls = self.url_lists.get(list_name, [])
        self.processor.process_urls(urls)
        self.processor.save_results()
        logging.info(f"--- Job {list_name} Finished ---")

    def _signal_handler(self, signum, frame):
        logging.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.stop()
        sys.exit(0)

    def stop(self):
        """Stops all components gracefully."""
        if self.scheduler.running: self.scheduler.shutdown()
        self.processor.shutdown()
        logging.info("--- Application Stopped ---")

def main():
    """Main entry point for the application."""
    config = ConfigManager()
    setup_logging(config)
    app = Application(config)
    try:
        app.run()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Shutdown initiated by user or system.")
    except Exception as e:
        logging.critical("An unhandled exception occurred!", exc_info=True)
    finally:
        app.stop()

if __name__ == "__main__":
    main()