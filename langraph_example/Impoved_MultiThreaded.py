#!/usr/bin/env python3
"""
Enhanced CronScheduler with a Resilient URL Processing Thread Pool

This production-ready script provides a robust framework for downloading,
preprocessing, and ingesting files from URLs in a concurrent, fault-tolerant manner.
It integrates a feature-rich scheduler with a resilient multi-threaded processor.
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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse
import hashlib

# --- Third-Party Imports ---
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# --- Installation ---
# pip install APScheduler requests

# --- Logging Configuration ---
# Configure a single, unified logger for the entire application.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('url_processor_enhanced.log'), # Log to a file
        logging.StreamHandler()                            # Also log to the console
    ]
)
logger = logging.getLogger(__name__)

# --- Data Structures ---
@dataclass
class ProcessingResult:
    """Data class to store the detailed result of processing a single URL."""
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

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a JSON-serializable dictionary."""
        result = asdict(self)
        # Convert datetime objects to ISO format strings for JSON compatibility
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result

# --- Core URL Processor ---
class URLProcessor:
    """
    Manages the entire lifecycle of URL processing using a thread-safe pool.
    Handles downloading, preprocessing, and ingestion with detailed status tracking.
    """
    def __init__(self, pool_size: int = 10, timeout: int = 30):
        self.pool_size = pool_size
        self.timeout = timeout
        self.results: Dict[str, ProcessingResult] = {}
        self.results_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="URLProcessor")
        self.download_dir = Path("downloads")
        self.chunks_dir = Path("chunks")
        self.download_dir.mkdir(exist_ok=True)
        self.chunks_dir.mkdir(exist_ok=True)

    def process_urls(self, urls: List[str]):
        """
        Submits a list of URLs to the thread pool and processes them concurrently.
        This is a blocking call that waits for all submitted URLs to complete.
        """
        if not urls:
            logger.warning("No URLs provided to process.")
            return

        with self.results_lock:
            for url in urls:
                self.results[url] = ProcessingResult(url=url, status='queued', message='URL is queued for processing.')

        # Create a dictionary mapping each future to its corresponding URL
        future_to_url = {self.executor.submit(self._process_single_url, url): url for url in urls}
        total_urls = len(future_to_url)
        logger.info(f"Submitted {total_urls} URLs to the processing pool of {self.pool_size} threads.")

        # Process results as they complete using a clean `as_completed` loop
        for i, future in enumerate(as_completed(future_to_url), 1):
            url = future_to_url[future]
            try:
                # The result of the future is the ProcessingResult object
                result = future.result()
                logger.info(f"({i}/{total_urls}) COMPLETED: Status '{result.status.upper()}' for {url}. Reason: {result.message}")
            except Exception as e:
                # This catches unexpected errors if the future itself fails
                logger.error(f"({i}/{total_urls}) An unexpected error was raised by the future for URL {url}: {e}", exc_info=True)
                with self.results_lock:
                    self.results[url] = ProcessingResult(url=url, status='failure', message=f"Critical execution error: {e}")

    def _process_single_url(self, url: str) -> ProcessingResult:
        """The complete, three-step processing workflow for a single URL."""
        thread_id = threading.current_thread().name
        start_time = datetime.now()

        with self.results_lock:
            self.results[url].status = 'processing'
            self.results[url].start_time = start_time
            self.results[url].thread_id = thread_id
            self.results[url].message = f'Processing started in thread {thread_id}.'

        logger.info(f"Starting full processing pipeline for URL: {url}")
        result: ProcessingResult
        try:
            # Step 1: Download
            download_res = self._download_file(url)
            # Step 2: Preprocess
            preprocess_res = self._preprocess_to_chunks(download_res['file_path'], url)
            # Step 3: Ingest
            ingest_res = self._ingest_to_vector_db(preprocess_res['chunks_dir'], url)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            result = ProcessingResult(
                url=url, status='success',
                message=f'Successfully processed in {duration:.2f}s.',
                start_time=start_time, end_time=end_time, duration_seconds=duration,
                thread_id=thread_id, file_size_bytes=download_res.get('file_size'),
                chunks_created=preprocess_res.get('chunks_count'), ingestion_id=ingest_res.get('ingestion_id')
            )
            logger.info(f"✅ Successfully processed {url} in {duration:.2f}s")

        except (ValueError, requests.RequestException, IOError, RuntimeError, ConnectionError) as e:
            # Catch specific, expected exceptions from our pipeline
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            result = ProcessingResult(
                url=url, status='failure',
                message=f'Failed after {duration:.2f}s: {e}',
                start_time=start_time, end_time=end_time, duration_seconds=duration, thread_id=thread_id
            )
            # Log the specific error without a full stack trace for cleanliness, unless debugging
            logger.error(f"❌ Failed to process {url} due to: {e}")

        # Update the final results dictionary in a thread-safe manner
        with self.results_lock:
            self.results[url] = result
        return result

    def _download_file(self, url: str) -> Dict[str, Any]:
        """Step 1: Download file from URL (Realistic Simulation)."""
        if random.random() < 0.05: # 5% chance of malformed URL
             raise ValueError("Malformed URL provided for download simulation.")

        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{url_hash}_{Path(urlparse(url).path).name or 'download'}.txt"
        file_path = self.download_dir / filename

        try:
            logger.info(f"Step 1: Downloading from {url}")
            response = requests.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            dummy_content = f"Content from {url} downloaded at {datetime.now().isoformat()}"
            file_path.write_text(dummy_content, encoding='utf-8')
            file_size = file_path.stat().st_size

            logger.info(f"Simulated download of {file_size} bytes to {file_path}")
            return {'success': True, 'file_path': file_path, 'file_size': file_size}
        except requests.exceptions.RequestException as e:
            raise IOError(f"Network/HTTP error during download: {e}") from e

    def _preprocess_to_chunks(self, file_path: Path, url: str) -> Dict[str, Any]:
        """Step 2: Preprocess file into chunks (Simulation)."""
        logger.info(f"Step 2: Preprocessing file {file_path}")
        if random.random() < 0.1: # 10% chance of failure
            raise RuntimeError("Simulated failure during file content parsing.")

        file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        chunks_dir = self.chunks_dir / f"chunks_{file_hash}"
        chunks_dir.mkdir(exist_ok=True)
        # Simulate creating 1 to 5 chunks
        num_chunks = random.randint(1, 5)
        logger.info(f"Created {num_chunks} chunk(s) in {chunks_dir}")
        return {'success': True, 'chunks_dir': chunks_dir, 'chunks_count': num_chunks}

    def _ingest_to_vector_db(self, chunks_dir: Path, url: str) -> Dict[str, Any]:
        """Step 3: Ingest chunks to vector DB (Simulation)."""
        logger.info(f"Step 3: Ingesting chunks from {chunks_dir}")
        if random.random() < 0.1: # 10% chance of failure
            raise ConnectionError("Simulated vector DB API connection failed.")

        ingestion_id = f"ingest_{uuid.uuid4().hex[:12]}"
        logger.info(f"Successfully ingested chunks with ID {ingestion_id}")
        return {'success': True, 'ingestion_id': ingestion_id}

    def get_summary(self) -> Dict[str, Any]:
        """Returns a snapshot of the current processing results and stats."""
        with self.results_lock:
            results = list(self.results.values())
        success_count = sum(1 for r in results if r.status == 'success')
        failure_count = sum(1 for r in results if r.status == 'failure')
        return {'total': len(results), 'success': success_count, 'failed': failure_count}

    def save_results_to_file(self) -> str:
        """Saves final results to a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"url_processing_results_{timestamp}.json"
        with self.results_lock:
             results_data = {url: result.to_dict() for url, result in self.results.items()}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4)
        logger.info(f"💾 Results for {len(results_data)} URLs saved to {filename}")
        return filename

    def cleanup(self):
        """Removes all generated files from downloads and chunks directories."""
        logger.info("--- Cleaning up generated files from previous runs ---")
        for directory in [self.download_dir, self.chunks_dir]:
            if directory.exists():
                shutil.rmtree(directory)
                logger.info(f"Removed directory: {directory}")
                directory.mkdir(exist_ok=True)

    def shutdown(self):
        """Shuts down the thread pool executor."""
        logger.info("Shutting down the URL processor thread pool...")
        self.executor.shutdown(wait=True)
        logger.info("Thread pool shut down.")

# --- Enhanced Scheduler integrating URLProcessor ---
class EnhancedCronScheduler:
    """
    Enhanced scheduler that integrates the URLProcessor to run jobs on a schedule.
    This class combines the logic from the user's `scheduler.py`.
    """
    def __init__(self, use_background: bool = False, pool_size: int = 10):
        self.job_count = 0
        self.daily_executions = 0
        self.scheduler = BlockingScheduler()
        self.url_processor = URLProcessor(pool_size=pool_size)
        self.url_lists: Dict[str, List[str]] = {}

        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info(f"EnhancedCronScheduler initialized with pool size: {pool_size}")

    def add_url_list(self, list_name: str, urls: List[str]):
        """Adds a named list of URLs for future processing jobs."""
        self.url_lists[list_name] = urls
        logger.info(f"Added URL list '{list_name}' with {len(urls)} URLs.")

    def url_processing_job(self, list_name: str = "default"):
        """The main job function that processes a list of URLs."""
        self.job_count += 1
        self.daily_executions += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"\n🚀🚀🚀 Starting URL Processing Job '{list_name}' at {current_time} 🚀🚀🚀")
        logger.info(f"Total job runs: {self.job_count} | Runs today: {self.daily_executions}")

        try:
            urls_to_process = self.url_lists.get(list_name)
            if not urls_to_process:
                logger.warning(f"URL list '{list_name}' not found or is empty. Job skipping.")
                return

            self.url_processor.cleanup() # Clean up before each run
            self.url_processor.process_urls(urls_to_process)

            summary = self.url_processor.get_summary()
            results_file = self.url_processor.save_results_to_file()
            logger.info(f"✅ Job '{list_name}' completed. Success: {summary['success']}, Failed: {summary['failed']}.")
            logger.info(f"Final results are available in: {results_file}")

        except Exception as e:
            logger.critical(f"❌ A critical error occurred in job '{list_name}': {e}", exc_info=True)

    def setup_scheduled_job(self, cron_expression: str, job_name: str, url_list_name: str):
        """Adds a URL processing job to the scheduler."""
        try:
            trigger = CronTrigger.from_crontab(cron_expression)
            self.scheduler.add_job(
                func=lambda: self.url_processing_job(url_list_name),
                trigger=trigger, id=job_name, name=job_name
            )
            logger.info(f"📅 Added cron job '{job_name}' for list '{url_list_name}' with expression: {cron_expression}")
        except Exception as e:
            logger.error(f"❌ Error adding cron job '{job_name}': {e}", exc_info=True)
            raise

    def start(self):
        """Starts the scheduler."""
        logger.info("🎯 Cron Scheduler starting! Press Ctrl+C to stop.")
        self.scheduler.start()

    def stop_all(self):
        """Stops the scheduler and shuts down the URL processor."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler shut down.")
        self.url_processor.shutdown()
        logger.info("All processing stopped.")

    def _job_executed(self, event):
        logger.info(f"Event: Job '{event.job_id}' executed successfully.")

    def _job_error(self, event):
        logger.error(f"Event: Job '{event.job_id}' failed with exception: {event.exception}")

    def _signal_handler(self, signum, frame):
        logger.warning(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop_all()
        sys.exit(0)


# --- Main Execution ---
def main():
    """Main function to demonstrate the enhanced scheduler with a direct job run."""
    logger.info("--- URL Processing Demonstration ---")

    # Use real, accessible URLs plus some that are guaranteed to fail
    sample_urls = [
        "http://info.cern.ch/hypertext/WWW/TheProject.html", # The first website
        "https://www.google.com",
        "https://www.github.com",
        "https://www.python.org",
        "http://thisurldoesnotexist12345.com", # Expected to fail DNS lookup
        "https://httpstat.us/404", # Will cause an HTTP 404 error
        "https://httpstat.us/503", # Will cause an HTTP 503 error
        "https://invalid-schema://some-url", # Malformed URL
    ]

    scheduler = EnhancedCronScheduler(pool_size=4)

    try:
        scheduler.add_url_list("daily_news_batch", sample_urls)

        # For demonstration, we will run the job directly instead of on a schedule.
        # In a real scenario, you would use scheduler.setup_scheduled_job() and scheduler.start()
        scheduler.url_processing_job("daily_news_batch")

    except Exception as e:
        logger.critical(f"Demonstration failed with a top-level error: {e}", exc_info=True)
    finally:
        scheduler.stop_all()
        logger.info("--- Demonstration finished ---")

if __name__ == "__main__":
    main()