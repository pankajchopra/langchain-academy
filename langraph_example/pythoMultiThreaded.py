#!/usr/bin/env python3
"""
Enhanced CronScheduler with URL Processing Thread Pool
A production-ready solution for downloading, preprocessing, and ingesting files from URLs
"""

import os
import json
import time
import logging
import threading
import uuid
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
from urllib.parse import urlparse
import hashlib

# Import the base scheduler
from scheduler import CronScheduler

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('url_processor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Data class to store processing results for each URL."""
    url: str
    status: str  # 'success', 'failure', 'processing', 'queued'
    message: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    thread_id: Optional[str] = None
    file_size: Optional[int] = None
    chunks_created: Optional[int] = None
    ingestion_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result


class URLProcessor:
    """
    Thread-safe URL processor that handles downloading, preprocessing, and ingestion.
    """
    
    def __init__(self, pool_size: int = 10, max_retries: int = 3, timeout: int = 300):
        """
        Initialize the URL processor.
        
        Args:
            pool_size (int): Number of worker threads in the pool
            max_retries (int): Maximum retry attempts for failed operations
            timeout (int): Timeout in seconds for each operation
        """
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Thread-safe data structures
        self.url_queue = Queue()
        self.results = {}  # Thread-safe dict for storing results
        self.results_lock = threading.RLock()
        
        # Thread pool
        self.executor = None
        self.is_running = False
        
        # Configuration
        self.download_dir = Path("downloads")
        self.chunks_dir = Path("chunks")
        self.download_dir.mkdir(exist_ok=True)
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_urls': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'in_progress': 0
        }
        self.stats_lock = threading.RLock()
    
    def add_urls(self, urls: List[str]) -> None:
        """
        Add URLs to the processing queue.
        
        Args:
            urls (List[str]): List of URLs to process
        """
        with self.stats_lock:
            self.stats['total_urls'] += len(urls)
        
        for url in urls:
            # Initialize result entry
            with self.results_lock:
                self.results[url] = ProcessingResult(
                    url=url,
                    status='queued',
                    message='URL queued for processing'
                )
            
            # Add to queue
            self.url_queue.put(url)
        
        logger.info(f"Added {len(urls)} URLs to processing queue")
    
    def start_processing(self) -> None:
        """Start the URL processing with thread pool."""
        if self.is_running:
            logger.warning("URL processor is already running")
            return
        
        self.is_running = True
        self.executor = ThreadPoolExecutor(
            max_workers=self.pool_size,
            thread_name_prefix="URLProcessor"
        )
        
        logger.info(f"Started URL processor with {self.pool_size} threads")
        
        # Submit initial batch of URLs
        self._submit_queued_urls()
    
    def stop_processing(self) -> None:
        """Stop the URL processing and clean up resources."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.executor:
            logger.info("Shutting down URL processor...")
            self.executor.shutdown(wait=True)
            self.executor = None
        
        logger.info("URL processor stopped")
    
    def _submit_queued_urls(self) -> None:
        """Submit queued URLs to the thread pool."""
        futures = []
        
        # Submit up to pool_size URLs initially
        submitted_count = 0
        while submitted_count < self.pool_size and not self.url_queue.empty():
            try:
                url = self.url_queue.get_nowait()
                future = self.executor.submit(self._process_single_url, url)
                futures.append(future)
                submitted_count += 1
                
                with self.stats_lock:
                    self.stats['in_progress'] += 1
                
            except Empty:
                break
        
        # Handle completed futures and submit new ones
        if futures:
            self._handle_completed_futures(futures)
    
    def _handle_completed_futures(self, initial_futures: List) -> None:
        """Handle completed futures and submit new URLs as threads become available."""
        active_futures = set(initial_futures)
        
        while active_futures and self.is_running:
            # Wait for at least one future to complete
            completed_futures = set()
            
            for future in as_completed(active_futures, timeout=1.0):
                completed_futures.add(future)
                
                with self.stats_lock:
                    self.stats['in_progress'] -= 1
                    self.stats['processed'] += 1
                
                # Submit a new URL if available
                if not self.url_queue.empty():
                    try:
                        url = self.url_queue.get_nowait()
                        new_future = self.executor.submit(self._process_single_url, url)
                        active_futures.add(new_future)
                        
                        with self.stats_lock:
                            self.stats['in_progress'] += 1
                        
                    except Empty:
                        pass
                
                # Remove completed future
                break
            
            # Remove completed futures from active set
            active_futures -= completed_futures
            
            # If no more URLs in queue and no active futures, we're done
            if self.url_queue.empty() and not active_futures:
                break
    
    def _process_single_url(self, url: str) -> ProcessingResult:
        """
        Process a single URL through the three-step workflow.
        
        Args:
            url (str): URL to process
            
        Returns:
            ProcessingResult: Result of the processing
        """
        thread_id = threading.current_thread().name
        start_time = datetime.now()
        
        # Update status to processing
        with self.results_lock:
            self.results[url].status = 'processing'
            self.results[url].start_time = start_time
            self.results[url].thread_id = thread_id
            self.results[url].message = f'Processing started in thread {thread_id}'
        
        logger.info(f"[{thread_id}] Starting processing of URL: {url}")
        
        try:
            # Step 1: Download the file
            logger.info(f"[{thread_id}] Step 1: Downloading from {url}")
            download_result = self._download_file(url, thread_id)
            
            if not download_result['success']:
                raise Exception(f"Download failed: {download_result['error']}")
            
            # Step 2: Preprocess into chunks
            logger.info(f"[{thread_id}] Step 2: Preprocessing file {download_result['file_path']}")
            preprocess_result = self._preprocess_to_chunks(
                download_result['file_path'], 
                url, 
                thread_id
            )
            
            if not preprocess_result['success']:
                raise Exception(f"Preprocessing failed: {preprocess_result['error']}")
            
            # Step 3: Ingest to vector database
            logger.info(f"[{thread_id}] Step 3: Ingesting chunks to vector database")
            ingest_result = self._ingest_to_vector_db(
                preprocess_result['chunks_dir'], 
                url, 
                thread_id
            )
            
            if not ingest_result['success']:
                raise Exception(f"Ingestion failed: {ingest_result['error']}")
            
            # Success!
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = ProcessingResult(
                url=url,
                status='success',
                message=f'Successfully processed in {duration:.2f}s - '
                       f'Downloaded: {download_result.get("file_size", 0)} bytes, '
                       f'Chunks: {preprocess_result.get("chunks_count", 0)}, '
                       f'Ingestion ID: {ingest_result.get("ingestion_id", "N/A")}',
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                thread_id=thread_id,
                file_size=download_result.get('file_size', 0),
                chunks_created=preprocess_result.get('chunks_count', 0),
                ingestion_id=ingest_result.get('ingestion_id')
            )
            
            with self.stats_lock:
                self.stats['successful'] += 1
            
            logger.info(f"[{thread_id}] ✅ Successfully processed {url} in {duration:.2f}s")
            
        except Exception as e:
            # Failure
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = ProcessingResult(
                url=url,
                status='failure',
                message=f'Failed after {duration:.2f}s: {str(e)}',
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                thread_id=thread_id
            )
            
            with self.stats_lock:
                self.stats['failed'] += 1
            
            logger.error(f"[{thread_id}] ❌ Failed to process {url}: {str(e)}")
        
        # Update final result
        with self.results_lock:
            self.results[url] = result
        
        return result
    
    def _download_file(self, url: str, thread_id: str) -> Dict[str, Any]:
        """
        Step 1: Download file from URL (DUMMY IMPLEMENTATION).
        
        Args:
            url (str): URL to download from
            thread_id (str): Thread identifier for logging
            
        Returns:
            Dict containing success status, file_path, file_size, or error
        """
        try:
            # Generate a unique filename based on URL
            parsed_url = urlparse(url)
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"{url_hash}_{parsed_url.path.split('/')[-1] or 'downloaded_file'}"
            if not any(filename.endswith(ext) for ext in ['.txt', '.json', '.csv', '.xml']):
                filename += '.txt'
            
            file_path = self.download_dir / filename
            
            # DUMMY IMPLEMENTATION: Simulate file download
            logger.info(f"[{thread_id}] Simulating download from {url}")
            
            # Simulate network delay
            time.sleep(0.5 + (hash(url) % 3))  # Random delay 0.5-3.5 seconds
            
            # Create dummy file content
            dummy_content = f"""# Downloaded from: {url}
# Downloaded at: {datetime.now().isoformat()}
# Thread: {thread_id}

This is dummy content for file downloaded from {url}.
Line 1: Sample data for processing
Line 2: More sample data with timestamp {datetime.now()}
Line 3: Additional content for chunking
Line 4: Document processing simulation
Line 5: Vector database ingestion test data

{"Sample JSON data": {"url": url, "thread": thread_id, "timestamp": datetime.now().isoformat()}}
"""
            
            # Write dummy content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(dummy_content)
            
            file_size = file_path.stat().st_size
            
            logger.info(f"[{thread_id}] ✅ Downloaded {file_size} bytes to {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'file_size': file_size,
                'filename': filename
            }
            
        except Exception as e:
            logger.error(f"[{thread_id}] ❌ Download failed for {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _preprocess_to_chunks(self, file_path: str, url: str, thread_id: str) -> Dict[str, Any]:
        """
        Step 2: Preprocess file into small .jsonl chunks (DUMMY IMPLEMENTATION).
        
        Args:
            file_path (str): Path to the downloaded file
            url (str): Original URL for reference
            thread_id (str): Thread identifier for logging
            
        Returns:
            Dict containing success status, chunks_dir, chunks_count, or error
        """
        try:
            # Create chunks directory for this file
            file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            chunks_dir = self.chunks_dir / f"chunks_{file_hash}"
            chunks_dir.mkdir(exist_ok=True)
            
            logger.info(f"[{thread_id}] Preprocessing file {file_path}")
            
            # DUMMY IMPLEMENTATION: Simulate file preprocessing
            time.sleep(0.3 + (hash(url) % 2))  # Random delay 0.3-2.3 seconds
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks (simulate chunking logic)
            lines = content.split('\n')
            chunk_size = 3  # Lines per chunk
            chunks_created = 0
            
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i+chunk_size]
                chunk_content = '\n'.join(chunk_lines)
                
                # Create JSONL format chunk
                chunk_data = {
                    'id': f"{file_hash}_chunk_{chunks_created}",
                    'source_url': url,
                    'chunk_index': chunks_created,
                    'content': chunk_content,
                    'metadata': {
                        'processed_at': datetime.now().isoformat(),
                        'thread_id': thread_id,
                        'original_file': file_path,
                        'chunk_size': len(chunk_content)
                    }
                }
                
                # Write chunk to .jsonl file
                chunk_file = chunks_dir / f"chunk_{chunks_created:04d}.jsonl"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False)
                    f.write('\n')
                
                chunks_created += 1
            
            logger.info(f"[{thread_id}] ✅ Created {chunks_created} chunks in {chunks_dir}")
            
            return {
                'success': True,
                'chunks_dir': str(chunks_dir),
                'chunks_count': chunks_created,
                'chunk_files': [str(chunks_dir / f"chunk_{i:04d}.jsonl") for i in range(chunks_created)]
            }
            
        except Exception as e:
            logger.error(f"[{thread_id}] ❌ Preprocessing failed for {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _ingest_to_vector_db(self, chunks_dir: str, url: str, thread_id: str) -> Dict[str, Any]:
        """
        Step 3: Ingest chunks to vector database via API (DUMMY IMPLEMENTATION).
        
        Args:
            chunks_dir (str): Directory containing the chunk files
            url (str): Original URL for reference
            thread_id (str): Thread identifier for logging
            
        Returns:
            Dict containing success status, ingestion_id, chunks_ingested, or error
        """
        try:
            chunks_path = Path(chunks_dir)
            chunk_files = list(chunks_path.glob("*.jsonl"))
            
            logger.info(f"[{thread_id}] Ingesting {len(chunk_files)} chunks to vector database")
            
            # DUMMY IMPLEMENTATION: Simulate vector database ingestion
            time.sleep(0.5 + (hash(url) % 2))  # Random delay 0.5-2.5 seconds
            
            # Simulate ingestion API call
            ingestion_id = f"ingest_{uuid.uuid4().hex[:12]}"
            chunks_ingested = 0
            
            for chunk_file in chunk_files:
                # Read chunk data
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                # Simulate API call to vector database
                # In real implementation, this would be:
                # response = requests.post(VECTOR_DB_INGEST_URL, json=chunk_data)
                
                # Simulate processing time
                time.sleep(0.1)
                
                chunks_ingested += 1
                
                logger.debug(f"[{thread_id}] Ingested chunk {chunk_data['id']}")
            
            # Simulate final ingestion confirmation
            ingestion_metadata = {
                'ingestion_id': ingestion_id,
                'source_url': url,
                'chunks_ingested': chunks_ingested,
                'ingested_at': datetime.now().isoformat(),
                'thread_id': thread_id,
                'status': 'completed'
            }
            
            # Save ingestion metadata
            metadata_file = chunks_path / "ingestion_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(ingestion_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{thread_id}] ✅ Successfully ingested {chunks_ingested} chunks with ID {ingestion_id}")
            
            return {
                'success': True,
                'ingestion_id': ingestion_id,
                'chunks_ingested': chunks_ingested,
                'metadata_file': str(metadata_file)
            }
            
        except Exception as e:
            logger.error(f"[{thread_id}] ❌ Ingestion failed for {chunks_dir}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current processing results.
        
        Returns:
            Dict mapping URLs to their processing results
        """
        with self.results_lock:
            return {url: result.to_dict() for url, result in self.results.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dict containing processing statistics
        """
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Add completion percentage
        if stats['total_urls'] > 0:
            stats['completion_percentage'] = (stats['processed'] / stats['total_urls']) * 100
        else:
            stats['completion_percentage'] = 0
        
        return stats
    
    def get_status_summary(self) -> str:
        """
        Get a human-readable status summary.
        
        Returns:
            String summary of current processing status
        """
        stats = self.get_statistics()
        
        return (f"📊 Processing Status: {stats['processed']}/{stats['total_urls']} URLs processed "
                f"({stats['completion_percentage']:.1f}% complete) | "
                f"✅ Success: {stats['successful']} | "
                f"❌ Failed: {stats['failed']} | "
                f"🔄 In Progress: {stats['in_progress']}")
    
    def save_results_to_file(self, filename: str = None) -> str:
        """
        Save processing results to a JSON file.
        
        Args:
            filename (str): Optional filename, defaults to timestamped file
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"url_processing_results_{timestamp}.json"
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'results': self.get_results()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to {filename}")
        return filename


class EnhancedCronScheduler(CronScheduler):
    """
    Enhanced CronScheduler with URL processing capabilities.
    """
    
    def __init__(self, use_background: bool = False, pool_size: int = 10):
        """
        Initialize enhanced scheduler with URL processing.
        
        Args:
            use_background (bool): Whether to use background scheduler
            pool_size (int): Thread pool size for URL processing
        """
        super().__init__(use_background)
        
        self.url_processor = URLProcessor(pool_size=pool_size)
        self.url_lists = {}  # Store different URL lists for different jobs
        
        logger.info(f"Enhanced CronScheduler initialized with pool size: {pool_size}")
    
    def add_url_list(self, list_name: str, urls: List[str]) -> None:
        """
        Add a named list of URLs for processing.
        
        Args:
            list_name (str): Name/identifier for this URL list
            urls (List[str]): List of URLs to process
        """
        self.url_lists[list_name] = urls
        logger.info(f"Added URL list '{list_name}' with {len(urls)} URLs")
    
    def url_processing_job(self, list_name: str = "default") -> None:
        """
        The scheduled job that processes URLs.
        This replaces your_scheduled_method for URL processing tasks.
        
        Args:
            list_name (str): Name of the URL list to process
        """
        self.job_count += 1
        self.daily_executions += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"🚀 URL Processing Job '{list_name}' started at {current_time}")
        print(f"📊 Total executions: {self.job_count} | Today: {self.daily_executions}")
        
        try:
            # Get URLs to process
            if list_name not in self.url_lists:
                raise ValueError(f"URL list '{list_name}' not found")
            
            urls = self.url_lists[list_name]
            
            if not urls:
                print(f"   ⚠️ No URLs found in list '{list_name}'")
                return
            
            print(f"   📋 Processing {len(urls)} URLs from list '{list_name}'")
            
            # Start URL processor if not running
            if not self.url_processor.is_running:
                self.url_processor.start_processing()
            
            # Add URLs to processor
            self.url_processor.add_urls(urls)
            
            # Wait for processing to complete
            print(f"   🔄 Processing URLs with {self.url_processor.pool_size} threads...")
            
            # Monitor progress
            while True:
                time.sleep(5)  # Check every 5 seconds
                stats = self.url_processor.get_statistics()
                
                print(f"   📈 {self.url_processor.get_status_summary()}")
                
                # Check if all URLs are processed
                if stats['processed'] >= stats['total_urls'] and stats['in_progress'] == 0:
                    break
            
            # Final results
            stats = self.url_processor.get_statistics()
            results_file = self.url_processor.save_results_to_file()
            
            success_message = (f"✅ URL processing completed! "
                             f"Processed: {stats['processed']}, "
                             f"Successful: {stats['successful']}, "
                             f"Failed: {stats['failed']} | "
                             f"Results saved to: {results_file}")
            
            print(f"   {success_message}")
            logging.info(f"URL processing job '{list_name}' completed successfully. {success_message}")
            
        except Exception as e:
            error_message = f"❌ Error in URL processing job '{list_name}': {e}"
            print(f"   {error_message}")
            logging.error(error_message)
            raise
        finally:
            # Clean up if needed
            pass
    
    def setup_url_processing_schedule(self, times: List[int] = None, url_list_name: str = "default"):
        """
        Set up scheduled URL processing job.
        
        Args:
            times (List[int]): Hours to run the job (defaults to [8, 14, 20])
            url_list_name (str): Name of URL list to process
        """
        if times is None:
            times = [8, 14, 20]
        
        # Create cron expression
        hours_str = ",".join(map(str, times))
        cron_expression = f"0 {hours_str} * * *"
        
        # Add the URL processing job
        job_method = lambda: self.url_processing_job(url_list_name)
        self.add_cron_job(cron_expression, f"url_processing_{url_list_name}", job_method)
        
        # Add daily reset job
        self.add_cron_job("1 0 * * *", "daily_reset", self.reset_daily_count)
        
        print(f"🎯 Configured URL processing for list '{url_list_name}' at hours: {times}")
        logger.info(f"URL processing scheduled for list '{url_list_name}' with cron: {cron_expression}")
        
        return cron_expression
    
    def stop_all_processing(self):
        """Stop all processing and clean up resources."""
        self.url_processor.stop_processing()
        self.stop_scheduler()
        logger.info("All processing stopped and resources cleaned up")


def main():
    """
    Main function demonstrating the enhanced scheduler with URL processing.
    """
    print("🚀 Enhanced CronScheduler with URL Processing")
    print("=" * 60)
    
    # Create enhanced scheduler
    scheduler = EnhancedCronScheduler(use_background=False, pool_size=5)
    
    # Example URLs for testing (replace with your actual URLs)
    sample_urls = [
        "https://example.com/file1.txt",
        "https://example.com/file2.json",
        "https://example.com/file3.csv",
        "https://example.com/file4.xml",
        "https://example.com/file5.txt",
        "https://example.com/file6.json",
        "https://example.com/file7.txt",
        "https://example.com/file8.csv",
    ]
    
    # Add URL lists
    scheduler.add_url_list("daily_batch", sample_urls[:4])
    scheduler.add_url_list("full_batch", sample_urls)
    
    # Setup scheduled processing (uncomment for production)
    # scheduler.setup_url_processing_schedule([9, 15, 21], "daily_batch")
    
    # For demonstration, run immediately
    print("\n🔄 Running immediate URL processing demonstration...")
    scheduler.url_processing_job("daily_batch")
    
    # Clean up
    scheduler.stop_all_processing()
    
    print("\n✅ Demonstration completed!")


def test_url_processor():
    """
    Test the URL processor independently.
    """
    print("🧪 Testing URL Processor")
    print("=" * 40)
    
    # Create processor
    processor = URLProcessor(pool_size=3)
    
    # Test URLs
    test_urls = [
        "https://test.com/file1.txt",
        "https://test.com/file2.json",
        "https://test.com/file3.csv",
        "https://test.com/file4.xml",
        "https://test.com/file5.txt",
    ]
    
    try:
        # Start processing
        processor.start_processing()
        
        # Add URLs
        processor.add_urls(test_urls)
        
        # Monitor progress
        while True:
            time.sleep(2)
            stats = processor.get_statistics()
            print(f"📊 {processor.get_status_summary()}")
            
            if stats['processed'] >= stats['total_urls'] and stats['in_progress'] == 0:
                break
        
        # Show final results
        results = processor.get_results()
        print(f"\n📋 Final Results:")
        for url, result in results.items():
            print(f"  {result['status'].upper()}: {url} - {result['message']}")
        
        # Save results
        results_file = processor.save_results_to_file()
        print(f"\n💾 Results saved to: {results_file}")
        
    finally:
        processor.stop_processing()


if __name__ == "__main__":
    # Choose which demonstration to run:
    
    # 1. Full enhanced scheduler demonstration
    main()
    
    # 2. URL processor test only (uncomment to use)
    # test_url_processor()


"""
INSTALLATION AND USAGE
=====================

Prerequisites:
pip install APScheduler requests

Key Features:
✅