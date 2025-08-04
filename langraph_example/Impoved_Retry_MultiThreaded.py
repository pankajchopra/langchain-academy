#!/usr/bin/env python3
"""
Production-Grade Enhanced CronScheduler with Resilient URL Processing

This enterprise-ready script provides a comprehensive framework for downloading,
preprocessing, and ingesting files from URLs with advanced features including:
- Exponential backoff retry logic
- External configuration management
- Comprehensive metrics and monitoring
- Health check endpoints
- Rate limiting and circuit breaker patterns
- Advanced logging and alerting
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
import asyncio
import math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from urllib.parse import urlparse
import hashlib
from collections import defaultdict, deque
from contextlib import contextmanager
import configparser
from threading import Event
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket

# --- Third-Party Imports ---
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# --- Installation ---
# pip install APScheduler requests

# --- Configuration Management ---
class ConfigManager:
    """Manages external configuration for the URL processor."""
    
    def __init__(self, config_file: str = "url_processor_config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._load_default_config()
        self._load_config_file()
    
    def _load_default_config(self):
        """Load default configuration values."""
        self.config.read_dict({
            'processing': {
                'pool_size': '10',
                'timeout': '30',
                'max_retries': '3',
                'base_retry_delay': '1.0',
                'max_retry_delay': '60.0',
                'rate_limit_requests_per_minute': '60',
                'circuit_breaker_failure_threshold': '5',
                'circuit_breaker_timeout': '300'
            },
            'paths': {
                'download_dir': 'downloads',
                'chunks_dir': 'chunks',
                'log_dir': 'logs',
                'results_dir': 'results',
                'config_dir': 'config'
            },
            'logging': {
                'level': 'INFO',
                'max_file_size': '10485760',  # 10MB
                'backup_count': '5',
                'format': '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
            },
            'monitoring': {
                'health_check_port': '8080',
                'metrics_enabled': 'true',
                'metrics_retention_days': '7',
                'alert_on_failure_rate': '0.2'
            },
            'database': {
                'vector_db_url': 'http://localhost:8000/api/ingest',
                'vector_db_timeout': '30',
                'vector_db_api_key': '',
                'batch_size': '10'
            }
        })
    
    def _load_config_file(self):
        """Load configuration from file if it exists."""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            logging.info(f"Configuration loaded from {self.config_file}")
        else:
            self._create_default_config_file()
    
    def _create_default_config_file(self):
        """Create a default configuration file."""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
        logging.info(f"Default configuration file created at {self.config_file}")
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get configuration value with type conversion."""
        value = self.config.get(section, key, fallback=str(fallback) if fallback is not None else None)
        
        # Type conversion based on common patterns
        if value is None:
            return fallback
        
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

# --- Enhanced Logging Configuration ---
def setup_advanced_logging(config: ConfigManager):
    """Set up advanced logging with rotation and multiple handlers."""
    from logging.handlers import RotatingFileHandler
    
    log_dir = Path(config.get('paths', 'log_dir'))
    log_dir.mkdir(exist_ok=True)
    
    # Main application log
    main_handler = RotatingFileHandler(
        log_dir / 'url_processor.log',
        maxBytes=config.get('logging', 'max_file_size'),
        backupCount=config.get('logging', 'backup_count')
    )
    
    # Error-only log
    error_handler = RotatingFileHandler(
        log_dir / 'url_processor_errors.log',
        maxBytes=config.get('logging', 'max_file_size'),
        backupCount=config.get('logging', 'backup_count')
    )
    error_handler.setLevel(logging.ERROR)
    
    # Performance metrics log
    metrics_handler = RotatingFileHandler(
        log_dir / 'url_processor_metrics.log',
        maxBytes=config.get('logging', 'max_file_size'),
        backupCount=config.get('logging', 'backup_count')
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    
    # Formatter
    formatter = logging.Formatter(config.get('logging', 'format'))
    for handler in [main_handler, error_handler, metrics_handler, console_handler]:
        handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.get('logging', 'level')),
        handlers=[main_handler, error_handler, console_handler]
    )
    
    # Separate metrics logger
    metrics_logger = logging.getLogger('metrics')
    metrics_logger.addHandler(metrics_handler)
    metrics_logger.setLevel(logging.INFO)
    
    return logging.getLogger(__name__), metrics_logger

# --- Data Structures ---
@dataclass
class ProcessingResult:
    """Enhanced data class to store detailed processing results."""
    url: str
    status: str
    message: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    thread_id: Optional[str] = None
    file_size_bytes: Optional[int] = None
    chunks_created: Optional[int] = None
    ingestion_id: Optional[str] = None
    retry_count: int = 0
    error_type: Optional[str] = None
    http_status_code: Optional[int] = None
    rate_limited: bool = False
    circuit_breaker_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a JSON-serializable dictionary."""
        result = asdict(self)
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result

@dataclass
class ProcessingMetrics:
    """Comprehensive metrics tracking."""
    total_urls: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    rate_limited: int = 0
    circuit_breaker_triggered: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    peak_processing_time: float = 0.0
    throughput_per_minute: float = 0.0
    error_breakdown: Dict[str, int] = field(default_factory=dict)
    hourly_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

# --- Rate Limiter ---
class RateLimiter:
    """Token bucket rate limiter for controlling request rates."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, timeout: float = 5.0) -> bool:
        """Acquire a token for rate limiting."""
        with self.lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(
                self.requests_per_minute,
                self.tokens + (elapsed * self.requests_per_minute / 60.0)
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            # Wait for token if timeout specified
            if timeout > 0:
                wait_time = min(timeout, 60.0 / self.requests_per_minute)
                time.sleep(wait_time)
                return self.acquire(timeout - wait_time)
            
            return False

# --- Circuit Breaker ---
class CircuitBreaker:
    """Circuit breaker pattern implementation for handling cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    @contextmanager
    def call(self):
        """Context manager for circuit breaker calls."""
        with self.lock:
            if self.state == 'OPEN':
                if self.last_failure_time and \
                   time.time() - self.last_failure_time >= self.timeout:
                    self.state = 'HALF_OPEN'
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                yield
                # Success
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                self.failure_count = 0
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                
                raise e

# --- Health Check Server ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""
    
    def __init__(self, url_processor, *args, **kwargs):
        self.url_processor = url_processor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for health checks."""
        if self.path == '/health':
            self._handle_health_check()
        elif self.path == '/metrics':
            self._handle_metrics()
        elif self.path == '/status':
            self._handle_status()
        else:
            self.send_error(404)
    
    def _handle_health_check(self):
        """Return basic health status."""
        health_status = {
            'status': 'healthy' if self.url_processor.is_healthy() else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.url_processor.start_time
        }
        
        self._send_json_response(health_status)
    
    def _handle_metrics(self):
        """Return processing metrics."""
        metrics = self.url_processor.get_metrics()
        self._send_json_response(asdict(metrics))
    
    def _handle_status(self):
        """Return detailed status information."""
        status = {
            'is_running': self.url_processor.is_running,
            'active_threads': threading.active_count(),
            'queue_size': self.url_processor.get_queue_size(),
            'circuit_breaker_state': self.url_processor.circuit_breaker.state,
            'rate_limiter_tokens': self.url_processor.rate_limiter.tokens
        }
        self._send_json_response(status)
    
    def _send_json_response(self, data):
        """Send JSON response."""
        json_data = json.dumps(data, indent=2)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data.encode())
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logging."""
        pass

class HealthCheckServer:
    """HTTP server for health check endpoints."""
    
    def __init__(self, url_processor, port: int = 8080):
        self.url_processor = url_processor
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start(self):
        """Start the health check server."""
        try:
            handler = lambda *args, **kwargs: HealthCheckHandler(self.url_processor, *args, **kwargs)
            self.server = HTTPServer(('localhost', self.port), handler)
            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                daemon=True,
                name="HealthCheckServer"
            )
            self.server_thread.start()
            logging.info(f"Health check server started on port {self.port}")
        except Exception as e:
            logging.error(f"Failed to start health check server: {e}")
    
    def stop(self):
        """Stop the health check server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.server_thread:
                self.server_thread.join()
            logging.info("Health check server stopped")

# --- Enhanced URL Processor ---
class EnhancedURLProcessor:
    """
    Production-grade URL processor with advanced features:
    - Exponential backoff retry logic
    - Rate limiting and circuit breaker
    - Comprehensive metrics and monitoring
    - Health checks and status endpoints
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.pool_size = config.get('processing', 'pool_size')
        self.timeout = config.get('processing', 'timeout')
        self.max_retries = config.get('processing', 'max_retries')
        self.base_retry_delay = config.get('processing', 'base_retry_delay')
        self.max_retry_delay = config.get('processing', 'max_retry_delay')
        
        # Threading
        self.results: Dict[str, ProcessingResult] = {}
        self.results_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(
            max_workers=self.pool_size, 
            thread_name_prefix="URLProcessor"
        )
        
        # Directories
        self.download_dir = Path(config.get('paths', 'download_dir'))
        self.chunks_dir = Path(config.get('paths', 'chunks_dir'))
        self.results_dir = Path(config.get('paths', 'results_dir'))
        for directory in [self.download_dir, self.chunks_dir, self.results_dir]:
            directory.mkdir(exist_ok=True)
        
        # Advanced features
        self.rate_limiter = RateLimiter(config.get('processing', 'rate_limit_requests_per_minute'))
        self.circuit_breaker = CircuitBreaker(
            config.get('processing', 'circuit_breaker_failure_threshold'),
            config.get('processing', 'circuit_breaker_timeout')
        )
        
        # Metrics and monitoring
        self.metrics = ProcessingMetrics()
        self.metrics_lock = threading.RLock()
        self.start_time = time.time()
        self.processing_times = deque(maxlen=1000)  # Keep last 1000 processing times
        
        # Health check server
        if config.get('monitoring', 'health_check_port'):
            self.health_server = HealthCheckServer(
                self, 
                config.get('monitoring', 'health_check_port')
            )
            self.health_server.start()
        
        # Shutdown event
        self.shutdown_event = Event()
        
        logging.info(f"Enhanced URL Processor initialized with {self.pool_size} threads")
    
    def process_urls(self, urls: List[str], priority: str = "normal"):
        """Process URLs with enhanced error handling and retry logic."""
        if not urls:
            logging.warning("No URLs provided to process.")
            return
        
        # Initialize results
        with self.results_lock:
            for url in urls:
                self.results[url] = ProcessingResult(
                    url=url, 
                    status='queued', 
                    message='URL is queued for processing.'
                )
        
        # Update metrics
        with self.metrics_lock:
            self.metrics.total_urls += len(urls)
        
        # Submit to thread pool
        future_to_url = {
            self.executor.submit(self._process_single_url_with_retry, url): url 
            for url in urls
        }
        
        total_urls = len(future_to_url)
        logging.info(f"Submitted {total_urls} URLs to processing pool")
        
        # Process results
        completed = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            completed += 1
            
            try:
                result = future.result()
                status_emoji = "✅" if result.status == 'success' else "❌"
                logging.info(
                    f"({completed}/{total_urls}) {status_emoji} {result.status.upper()}: "
                    f"{url} - {result.message}"
                )
                
                # Log metrics
                if self.config.get('monitoring', 'metrics_enabled'):
                    self._log_processing_metrics(result)
                
            except Exception as e:
                logging.error(f"({completed}/{total_urls}) CRITICAL ERROR for {url}: {e}", exc_info=True)
                with self.results_lock:
                    self.results[url] = ProcessingResult(
                        url=url, 
                        status='failure', 
                        message=f"Critical execution error: {e}",
                        error_type='CRITICAL_ERROR'
                    )
    
    def _process_single_url_with_retry(self, url: str) -> ProcessingResult:
        """Process single URL with exponential backoff retry logic."""
        thread_id = threading.current_thread().name
        start_time = datetime.now()
        
        with self.results_lock:
            self.results[url].status = 'processing'
            self.results[url].start_time = start_time
            self.results[url].thread_id = thread_id
        
        retry_count = 0
        last_exception = None
        
        while retry_count <= self.max_retries:
            try:
                # Check for shutdown
                if self.shutdown_event.is_set():
                    raise Exception("Processor shutdown requested")
                
                # Rate limiting
                if not self.rate_limiter.acquire():
                    with self.results_lock:
                        self.results[url].rate_limited = True
                    with self.metrics_lock:
                        self.metrics.rate_limited += 1
                    raise Exception("Rate limit exceeded")
                
                # Circuit breaker
                with self.circuit_breaker.call():
                    result = self._process_single_url(url, retry_count)
                    
                    # Success - update metrics
                    with self.metrics_lock:
                        self.metrics.successful += 1
                        if retry_count > 0:
                            self.metrics.retried += 1
                    
                    return result
                    
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Check if circuit breaker was triggered
                if self.circuit_breaker.state == 'OPEN':
                    with self.results_lock:
                        self.results[url].circuit_breaker_triggered = True
                    with self.metrics_lock:
                        self.metrics.circuit_breaker_triggered += 1
                
                if retry_count <= self.max_retries:
                    # Calculate exponential backoff delay
                    delay = min(
                        self.base_retry_delay * (2 ** (retry_count - 1)),
                        self.max_retry_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.1) * delay
                    total_delay = delay + jitter
                    
                    logging.warning(
                        f"[{thread_id}] Retry {retry_count}/{self.max_retries} for {url} "
                        f"after {total_delay:.2f}s delay. Error: {e}"
                    )
                    
                    time.sleep(total_delay)
                else:
                    # Final failure
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    error_type = type(last_exception).__name__
                    with self.metrics_lock:
                        self.metrics.failed += 1
                        self.metrics.error_breakdown[error_type] = \
                            self.metrics.error_breakdown.get(error_type, 0) + 1
                    
                    result = ProcessingResult(
                        url=url,
                        status='failure',
                        message=f'Failed after {retry_count} retries in {duration:.2f}s: {last_exception}',
                        start_time=start_time,
                        end_time=end_time,
                        duration_seconds=duration,
                        thread_id=thread_id,
                        retry_count=retry_count,
                        error_type=error_type
                    )
                    
                    with self.results_lock:
                        self.results[url] = result
                    
                    return result
        
        # Should never reach here, but just in case
        return ProcessingResult(
            url=url,
            status='failure',
            message='Unexpected retry loop exit',
            error_type='RETRY_LOOP_ERROR'
        )
    
    def _process_single_url(self, url: str, retry_count: int) -> ProcessingResult:
        """Core URL processing logic (enhanced from original)."""
        thread_id = threading.current_thread().name
        start_time = datetime.now()
        
        logging.info(f"[{thread_id}] Processing {url} (attempt {retry_count + 1})")
        
        try:
            # Step 1: Download
            download_res = self._download_file(url)
            
            # Step 2: Preprocess
            preprocess_res = self._preprocess_to_chunks(download_res['file_path'], url)
            
            # Step 3: Ingest
            ingest_res = self._ingest_to_vector_db(preprocess_res['chunks_dir'], url)
            
            # Success
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Track processing time for metrics
            with self.metrics_lock:
                self.processing_times.append(duration)
                self.metrics.total_processing_time += duration
                self.metrics.peak_processing_time = max(
                    self.metrics.peak_processing_time, duration
                )
            
            result = ProcessingResult(
                url=url,
                status='success',
                message=f'Successfully processed in {duration:.2f}s',
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                thread_id=thread_id,
                file_size_bytes=download_res.get('file_size'),
                chunks_created=preprocess_res.get('chunks_count'),
                ingestion_id=ingest_res.get('ingestion_id'),
                retry_count=retry_count
            )
            
            with self.results_lock:
                self.results[url] = result
            
            return result
            
        except requests.RequestException as e:
            # Network-related errors - likely transient
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code == 429:  # Rate limited
                    raise Exception(f"Rate limited by server (HTTP {status_code})")
                elif 500 <= status_code < 600:  # Server errors - transient
                    raise Exception(f"Server error (HTTP {status_code}): {e}")
            raise Exception(f"Network error: {e}")
            
        except (IOError, OSError) as e:
            # File system errors - may be transient
            raise Exception(f"File system error: {e}")
            
        except Exception as e:
            # Other errors - likely permanent
            raise Exception(f"Processing error: {e}")
    
    def _download_file(self, url: str) -> Dict[str, Any]:
        """Enhanced download with better error handling and validation."""
        # Simulate failures for testing
        if random.random() < 0.05:  # 5% chance
            raise requests.RequestException("Simulated network timeout")
        
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        parsed_url = urlparse(url)
        filename = f"{url_hash}_{Path(parsed_url.path).name or 'download'}.txt"
        file_path = self.download_dir / filename
        
        try:
            # Simulate HTTP request with realistic behavior
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Enhanced-URL-Processor/1.0',
                'Accept': '*/*'
            })
            
            response = session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Create realistic dummy content
            dummy_content = f"""# Downloaded from: {url}
# Downloaded at: {datetime.now().isoformat()}
# Content-Type: {response.headers.get('content-type', 'unknown')}
# Status Code: {response.status_code}

This is enhanced dummy content for file downloaded from {url}.
Processing timestamp: {datetime.now()}
Thread: {threading.current_thread().name}

{'Sample data line ' + str(i) for i in range(10)}

Metadata:
- URL: {url}
- File size: {len(dummy_content)} bytes
- Processing session: {uuid.uuid4()}
"""
            
            file_path.write_text(dummy_content, encoding='utf-8')
            file_size = file_path.stat().st_size
            
            logging.info(f"Downloaded {file_size} bytes from {url} to {file_path}")
            
            return {
                'success': True,
                'file_path': file_path,
                'file_size': file_size,
                'content_type': response.headers.get('content-type', 'unknown'),
                'status_code': response.status_code
            }
            
        except requests.exceptions.Timeout:
            raise requests.RequestException(f"Request timeout after {self.timeout}s for {url}")
        except requests.exceptions.ConnectionError as e:
            raise requests.RequestException(f"Connection error for {url}: {e}")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 'unknown'
            raise requests.RequestException(f"HTTP {status_code} error for {url}: {e}")
        except Exception as e:
            raise IOError(f"Download failed for {url}: {e}")
    
    def _preprocess_to_chunks(self, file_path: Path, url: str) -> Dict[str, Any]:
        """Enhanced preprocessing with better chunk management."""
        if random.random() < 0.08:  # 8% chance
            raise RuntimeError("Simulated preprocessing failure")
        
        file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        chunks_dir = self.chunks_dir / f"chunks_{file_hash}_{int(time.time())}"
        chunks_dir.mkdir(exist_ok=True)
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Enhanced chunking logic
            lines = content.split('\n')
            chunk_size = self.config.get('processing', 'chunk_size', 5)  # Default 5 lines per chunk
            chunks_created = 0
            
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i+chunk_size]
                chunk_content = '\n'.join(chunk_lines)
                
                if not chunk_content.strip():  # Skip empty chunks
                    continue
                
                chunk_data = {
                    'id': f"{file_hash}_chunk_{chunks_created:04d}",
                    'source_url': url,
                    'chunk_index': chunks_created,
                    'content': chunk_content,
                    'metadata': {
                        'processed_at': datetime.now().isoformat(),
                        'thread_id': threading.current_thread().name,
                        'original_file': str(file_path),
                        'chunk_size_chars': len(chunk_content),
                        'chunk_size_lines': len(chunk_lines),
                        'total_chunks': math.ceil(len(lines) / chunk_size)
                    }
                }
                
                chunk_file = chunks_dir / f"chunk_{chunks_created:04d}.jsonl"
                chunk_file.write_text(json.dumps(chunk_data, ensure_ascii=False) + '\n', encoding='utf-8')
                chunks_created += 1
            
            logging.info(f"Created {chunks_created} chunks in {chunks_dir}")
            
            return {
                'success': True,
                'chunks_dir': chunks_dir,
                'chunks_count': chunks_created,
                'total_size_chars': len(content)
            }
            
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed for {file_path}: {e}")
    
    def _ingest_to_vector_db(self, chunks_dir: Path, url: str) -> Dict[str, Any]:
        """Enhanced vector DB ingestion with batch processing."""
        if random.random() < 0.06:  # 6% chance
            raise ConnectionError("Simulated vector DB connection failure")
        
        chunk_files = list(chunks_dir.glob("*.jsonl"))
        if not chunk_files: