#!/usr/bin/env python3
"""
Comprehensive Unit Tests for CronScheduler class from scheduler.py
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import time
import signal
import sys
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.job import Job
from apscheduler.jobstores.base import JobLookupError

# Import the actual CronScheduler class
from scheduler import CronScheduler


class TestCronSchedulerInitialization(unittest.TestCase):
    """Test cases for CronScheduler initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch signal.signal to prevent actual signal handler registration during tests
        self.signal_patcher = patch('scheduler.signal.signal')
        self.mock_signal = self.signal_patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.signal_patcher.stop()
    
    def test_init_blocking_scheduler_default(self):
        """Test initialization with default blocking scheduler."""
        scheduler = CronScheduler()
        
        self.assertIsInstance(scheduler.scheduler, BlockingScheduler)
        self.assertEqual(scheduler.job_count, 0)
        self.assertEqual(scheduler.daily_executions, 0)
        self.assertFalse(scheduler.use_background)
        
        # Verify signal handlers were registered
        expected_calls = [
            call(signal.SIGINT, scheduler._signal_handler),
            call(signal.SIGTERM, scheduler._signal_handler)
        ]
        self.mock_signal.assert_has_calls(expected_calls, any_order=True)
    
    def test_init_blocking_scheduler_explicit(self):
        """Test initialization with explicit blocking scheduler."""
        scheduler = CronScheduler(use_background=False)
        
        self.assertIsInstance(scheduler.scheduler, BlockingScheduler)
        self.assertFalse(scheduler.use_background)
    
    def test_init_background_scheduler(self):
        """Test initialization with background scheduler."""
        scheduler = CronScheduler(use_background=True)
        
        self.assertIsInstance(scheduler.scheduler, BackgroundScheduler)
        self.assertTrue(scheduler.use_background)
        self.assertEqual(scheduler.job_count, 0)
        self.assertEqual(scheduler.daily_executions, 0)
    
    @patch('scheduler.logging.basicConfig')
    def test_logging_configuration(self, mock_logging_config):
        """Test that logging is configured properly on import."""
        # Import should trigger logging configuration
        import scheduler
        
        # Verify logging was configured (called during module import)
        mock_logging_config.assert_called()


class TestScheduledMethod(unittest.TestCase):
    """Test cases for the your_scheduled_method function."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    @patch('scheduler.time.sleep')
    @patch('scheduler.logging.info')
    @patch('builtins.print')
    def test_your_scheduled_method_default_name(self, mock_print, mock_log_info, mock_sleep):
        """Test scheduled method with default job name."""
        initial_job_count = self.scheduler.job_count
        initial_daily_count = self.scheduler.daily_executions
        
        self.scheduler.your_scheduled_method()
        
        # Verify counters incremented
        self.assertEqual(self.scheduler.job_count, initial_job_count + 1)
        self.assertEqual(self.scheduler.daily_executions, initial_daily_count + 1)
        
        # Verify print statements were called
        self.assertTrue(mock_print.called)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("default" in call for call in print_calls))
        
        # Verify logging
        mock_log_info.assert_called()
        
        # Verify sleep was called (simulating work)
        mock_sleep.assert_called_once_with(1)
    
    @patch('scheduler.time.sleep')
    @patch('builtins.print')
    def test_your_scheduled_method_custom_name(self, mock_print, mock_sleep):
        """Test scheduled method with custom job name."""
        job_name = "test_custom_job"
        
        self.scheduler.your_scheduled_method(job_name)
        
        # Verify counters incremented
        self.assertEqual(self.scheduler.job_count, 1)
        self.assertEqual(self.scheduler.daily_executions, 1)
        
        # Verify custom job name appears in output
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any(job_name in call for call in print_calls))
    
    @patch('scheduler.time.sleep')
    @patch('scheduler.logging.error')
    @patch('builtins.print')
    def test_your_scheduled_method_with_exception(self, mock_print, mock_log_error, mock_sleep):
        """Test scheduled method exception handling."""
        mock_sleep.side_effect = Exception("Simulated error")
        
        with self.assertRaises(Exception) as context:
            self.scheduler.your_scheduled_method("error_job")
        
        self.assertIn("Simulated error", str(context.exception))
        
        # Verify counters still incremented (before exception)
        self.assertEqual(self.scheduler.job_count, 1)
        self.assertEqual(self.scheduler.daily_executions, 1)
        
        # Verify error was logged
        mock_log_error.assert_called()
        
        # Verify error message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("❌" in call and "error_job" in call for call in print_calls))
    
    def test_multiple_method_calls_increment_counters(self):
        """Test multiple calls increment counters correctly."""
        with patch('scheduler.time.sleep'), patch('builtins.print'), patch('scheduler.logging.info'):
            self.scheduler.your_scheduled_method("job1")
            self.scheduler.your_scheduled_method("job2")
            self.scheduler.your_scheduled_method("job3")
        
        self.assertEqual(self.scheduler.job_count, 3)
        self.assertEqual(self.scheduler.daily_executions, 3)


class TestCronJobManagement(unittest.TestCase):
    """Test cases for cron job management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    @patch('scheduler.logging.info')
    @patch('builtins.print')
    def test_add_cron_job_valid_expression(self, mock_print, mock_log_info):
        """Test adding a job with valid cron expression."""
        cron_expr = "0 8,14,20 * * *"
        job_name = "test_job"
        
        self.scheduler.add_cron_job(cron_expr, job_name)
        
        # Verify job was added
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].id, job_name)
        self.assertEqual(jobs[0].name, job_name)
        
        # Verify logging and printing
        mock_log_info.assert_called()
        mock_print.assert_called()
    
    def test_add_cron_job_invalid_expression(self):
        """Test adding a job with invalid cron expression."""
        invalid_expressions = [
            "invalid cron",
            "60 * * * *",  # Invalid minute
            "* 25 * * *",  # Invalid hour
            "",            # Empty string
            "* * * *",     # Too few fields
        ]
        
        for invalid_expr in invalid_expressions:
            with self.assertRaises((ValueError, TypeError)):
                self.scheduler.add_cron_job(invalid_expr, "invalid_job")
    
    @patch('builtins.print')
    def test_add_cron_job_with_custom_method(self, mock_print):
        """Test adding a job with custom method."""
        mock_method = Mock(return_value="custom result")
        cron_expr = "0 12 * * *"
        job_name = "custom_job"
        
        self.scheduler.add_cron_job(cron_expr, job_name, mock_method)
        
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].id, job_name)
        
        # Test that the custom method would be called (can't actually trigger APScheduler in tests)
        self.assertEqual(jobs[0].func, mock_method)
    
    def test_add_cron_job_with_kwargs(self):
        """Test adding a job with additional keyword arguments."""
        cron_expr = "0 9 * * *"
        job_name = "kwargs_job"
        
        # APScheduler kwargs
        self.scheduler.add_cron_job(
            cron_expr, 
            job_name, 
            max_instances=2,
            coalesce=True
        )
        
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.max_instances, 2)
        self.assertTrue(job.coalesce)
    
    @patch('scheduler.logging.error')
    @patch('builtins.print')
    def test_add_cron_job_duplicate_id_replaces(self, mock_print, mock_log_error):
        """Test that adding a job with duplicate ID replaces the existing job."""
        cron_expr1 = "0 9 * * *"
        cron_expr2 = "0 17 * * *"
        job_name = "duplicate_job"
        
        # Add first job
        self.scheduler.add_cron_job(cron_expr1, job_name)
        self.assertEqual(len(self.scheduler.scheduler.get_jobs()), 1)
        
        # Add second job with same name (should replace)
        self.scheduler.add_cron_job(cron_expr2, job_name)
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 1)
        
        # Verify the job was replaced by checking the trigger
        job = jobs[0]
        self.assertEqual(job.id, job_name)


class TestThreeDailyJobsSetup(unittest.TestCase):
    """Test cases for setup_three_daily_jobs method."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    @patch('builtins.print')
    def test_setup_three_daily_jobs_default_times(self, mock_print):
        """Test setup with default times."""
        cron_expr = self.scheduler.setup_three_daily_jobs()
        
        self.assertEqual(cron_expr, "0 8,14,20 * * *")
        
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 2)  # Main job + daily reset job
        
        job_ids = [job.id for job in jobs]
        self.assertIn("three_daily_executions", job_ids)
        self.assertIn("daily_reset", job_ids)
    
    @patch('builtins.print')
    def test_setup_three_daily_jobs_custom_times(self, mock_print):
        """Test setup with custom times."""
        custom_times = [6, 12, 18]
        cron_expr = self.scheduler.setup_three_daily_jobs(custom_times)
        
        self.assertEqual(cron_expr, "0 6,12,18 * * *")
        
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 2)
    
    def test_setup_three_daily_jobs_invalid_count(self):
        """Test setup with invalid number of times."""
        invalid_counts = [
            [8],              # Only 1 time
            [8, 14],          # Only 2 times  
            [8, 14, 20, 22],  # 4 times
            [],               # Empty list
        ]
        
        for invalid_times in invalid_counts:
            with self.assertRaises(ValueError) as context:
                self.scheduler.setup_three_daily_jobs(invalid_times)
            self.assertIn("Exactly 3 times", str(context.exception))
    
    def test_setup_three_daily_jobs_invalid_hours(self):
        """Test setup with invalid hour values."""
        invalid_hours_list = [
            [8, 14, 25],    # 25 is invalid (>23)
            [-1, 14, 20],   # -1 is invalid (<0)
            [8, 24, 20],    # 24 is invalid (should be 0-23)
            [8, 14, -5],    # -5 is invalid
        ]
        
        for invalid_hours in invalid_hours_list:
            with self.assertRaises(ValueError) as context:
                self.scheduler.setup_three_daily_jobs(invalid_hours)
            self.assertIn("must be between 0 and 23", str(context.exception))
    
    @patch('builtins.print')
    def test_setup_three_daily_jobs_edge_case_hours(self, mock_print):
        """Test setup with edge case valid hours."""
        edge_cases = [
            [0, 12, 23],    # Midnight, noon, 11 PM
            [1, 2, 3],      # Early morning hours
            [21, 22, 23],   # Late evening hours
        ]
        
        for times in edge_cases:
            scheduler = CronScheduler()
            with patch('scheduler.signal.signal'):
                cron_expr = scheduler.setup_three_daily_jobs(times)
                
                expected_expr = f"0 {','.join(map(str, times))} * * *"
                self.assertEqual(cron_expr, expected_expr)
                
                jobs = scheduler.scheduler.get_jobs()
                self.assertEqual(len(jobs), 2)


class TestJobManagementOperations(unittest.TestCase):
    """Test cases for job management operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    @patch('builtins.print')
    def test_list_jobs_empty(self, mock_print):
        """Test listing jobs when no jobs are scheduled."""
        self.scheduler.list_jobs()
        
        # Verify appropriate message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("No jobs currently scheduled" in call for call in print_calls))
    
    @patch('builtins.print')
    def test_list_jobs_with_jobs(self, mock_print):
        """Test listing jobs when jobs are scheduled."""
        # Add some jobs
        self.scheduler.add_cron_job("0 9 * * *", "morning_job")
        self.scheduler.add_cron_job("0 17 * * *", "evening_job")
        
        self.scheduler.list_jobs()
        
        # Verify job information was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        print_output = " ".join(print_calls)
        
        self.assertIn("morning_job", print_output)
        self.assertIn("evening_job", print_output)
        self.assertIn("Current scheduled jobs", print_output)
    
    @patch('scheduler.logging.info')
    @patch('builtins.print')
    def test_remove_job_existing(self, mock_print, mock_log_info):
        """Test removing an existing job."""
        job_name = "removable_job"
        self.scheduler.add_cron_job("0 12 * * *", job_name)
        
        # Verify job exists
        self.assertEqual(len(self.scheduler.scheduler.get_jobs()), 1)
        
        # Remove the job
        self.scheduler.remove_job(job_name)
        
        # Verify job was removed
        self.assertEqual(len(self.scheduler.scheduler.get_jobs()), 0)
        
        # Verify logging and printing
        mock_log_info.assert_called()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Removed job" in call and job_name in call for call in print_calls))
    
    @patch('scheduler.logging.error')
    @patch('builtins.print')
    def test_remove_job_nonexistent(self, mock_print, mock_log_error):
        """Test removing a non-existent job."""
        nonexistent_job = "nonexistent_job"
        
        self.scheduler.remove_job(nonexistent_job)
        
        # Verify error was logged and printed
        mock_log_error.assert_called()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Error removing job" in call and nonexistent_job in call for call in print_calls))


class TestDailyCountOperations(unittest.TestCase):
    """Test cases for daily count operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    @patch('scheduler.logging.info')
    @patch('builtins.print')
    def test_reset_daily_count(self, mock_print, mock_log_info):
        """Test resetting daily execution count."""
        # Set some initial values
        self.scheduler.daily_executions = 5
        self.scheduler.job_count = 10
        
        self.scheduler.reset_daily_count()
        
        # Verify only daily count was reset
        self.assertEqual(self.scheduler.daily_executions, 0)
        self.assertEqual(self.scheduler.job_count, 10)  # Should remain unchanged
        
        # Verify logging and printing
        mock_log_info.assert_called()
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Daily execution counter reset" in call for call in print_calls))


class TestCustomJobsPlaceholder(unittest.TestCase):
    """Test cases for add_custom_jobs method."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    def test_add_custom_jobs_default_empty(self):
        """Test that add_custom_jobs does nothing by default."""
        initial_job_count = len(self.scheduler.scheduler.get_jobs())
        
        self.scheduler.add_custom_jobs()
        
        # Should not add any jobs
        final_job_count = len(self.scheduler.scheduler.get_jobs())
        self.assertEqual(initial_job_count, final_job_count)


class TestSchedulerLifecycle(unittest.TestCase):
    """Test cases for scheduler lifecycle management."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    def test_stop_scheduler_when_not_running(self):
        """Test stopping scheduler when it's not running."""
        # Should not raise an exception
        with patch('builtins.print') as mock_print:
            self.scheduler.stop_scheduler()
            
            # Should not print anything since scheduler wasn't running
            print_calls = [str(call) for call in mock_print.call_args_list]
            stopped_messages = [call for call in print_calls if "stopped gracefully" in call]
            self.assertEqual(len(stopped_messages), 0)
    
    @patch('builtins.print')
    def test_start_scheduler_background_mode(self, mock_print):
        """Test starting scheduler in background mode."""
        scheduler = CronScheduler(use_background=True)
        with patch('scheduler.signal.signal'):
            # Add a job to test
            scheduler.add_cron_job("0 12 * * *", "test_job")
            
            # Mock the scheduler start and sleep to avoid actual blocking
            with patch.object(scheduler.scheduler, 'start'), \
                 patch.object(scheduler.scheduler, 'running', True), \
                 patch('scheduler.time.sleep', side_effect=KeyboardInterrupt):
                
                scheduler.start_scheduler()
                
                # Verify appropriate messages were printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                self.assertTrue(any("Background" in call for call in print_calls))


class TestEventListeners(unittest.TestCase):
    """Test cases for event listener functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    @patch('scheduler.logging.info')
    def test_job_executed_listener(self, mock_log_info):
        """Test job executed event listener."""
        # Create mock event
        mock_event = Mock()
        mock_event.job_id = "test_job"
        
        self.scheduler._job_executed(mock_event)
        
        # Verify logging was called
        mock_log_info.assert_called_once()
        args = mock_log_info.call_args[0]
        self.assertIn("test_job", args[0])
        self.assertIn("executed successfully", args[0])
    
    @patch('scheduler.logging.error')
    @patch('builtins.print')
    def test_job_error_listener(self, mock_print, mock_log_error):
        """Test job error event listener."""
        # Create mock event
        mock_event = Mock()
        mock_event.job_id = "error_job"
        mock_event.exception = Exception("Test error")
        
        self.scheduler._job_error(mock_event)
        
        # Verify error logging and printing
        mock_log_error.assert_called_once()
        mock_print.assert_called_once()
        
        # Check log message content
        log_args = mock_log_error.call_args[0]
        self.assertIn("error_job", log_args[0])
        self.assertIn("failed", log_args[0])


class TestSignalHandling(unittest.TestCase):
    """Test cases for signal handling."""
    
    @patch('scheduler.sys.exit')
    @patch('builtins.print')
    def test_signal_handler_execution(self, mock_print, mock_exit):
        """Test signal handler execution."""
        with patch('scheduler.signal.signal'):
            scheduler = CronScheduler()
            
            # Mock the stop_scheduler method
            with patch.object(scheduler, 'stop_scheduler') as mock_stop:
                scheduler._signal_handler(signal.SIGINT, None)
                
                # Verify stop_scheduler was called
                mock_stop.assert_called_once()
                
                # Verify sys.exit was called
                mock_exit.assert_called_once_with(0)
                
                # Verify message was printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                self.assertTrue(any("shutting down gracefully" in call for call in print_calls))


class TestValidCronExpressions(unittest.TestCase):
    """Test cases for various valid cron expressions."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    def test_common_cron_expressions(self):
        """Test commonly used cron expressions."""
        valid_expressions = [
            ("0 8,14,20 * * *", "three_times_daily"),
            ("*/15 * * * *", "every_15_minutes"),
            ("0 9-17 * * 1-5", "business_hours_weekdays"),
            ("0 0 1 * *", "monthly_first_day"),
            ("0 6 * * 1", "monday_6am"),
            ("30 */4 * * *", "every_4_hours_at_30"),
            ("0,30 * * * *", "twice_hourly"),
            ("15 2 * * *", "daily_2_15am"),
            ("0 12 * * 0", "sunday_noon"),
            ("45 23 * * 6", "saturday_11_45pm"),
        ]
        
        for cron_expr, job_name in valid_expressions:
            with self.subTest(cron_expr=cron_expr):
                # Clear previous jobs
                for job in self.scheduler.scheduler.get_jobs():
                    self.scheduler.scheduler.remove_job(job.id)
                
                # Should not raise an exception
                self.scheduler.add_cron_job(cron_expr, job_name)
                
                # Verify job was added
                jobs = self.scheduler.scheduler.get_jobs()
                self.assertEqual(len(jobs), 1)
                self.assertEqual(jobs[0].id, job_name)


class TestIntegrationWorkflows(unittest.TestCase):
    """Integration test cases for complete workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    @patch('builtins.print')
    def test_complete_setup_workflow(self, mock_print):
        """Test complete setup workflow."""
        # Setup three daily jobs
        cron_expr = self.scheduler.setup_three_daily_jobs([9, 15, 21])
        
        # Add custom jobs
        self.scheduler.add_cron_job("*/30 * * * *", "half_hourly_check")
        self.scheduler.add_cron_job("0 0 * * 1", "weekly_monday")
        
        # List all jobs
        self.scheduler.list_jobs()
        
        # Verify all jobs are present
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 4)  # 3 daily + daily reset + 2 custom
        
        job_ids = [job.id for job in jobs]
        expected_jobs = ["three_daily_executions", "daily_reset", "half_hourly_check", "weekly_monday"]
        for expected_job in expected_jobs:
            self.assertIn(expected_job, job_ids)
    
    @patch('scheduler.time.sleep')
    @patch('builtins.print')
    def test_job_execution_and_counting(self, mock_print, mock_sleep):
        """Test job execution and counter management."""
        # Execute jobs multiple times
        for i in range(5):
            self.scheduler.your_scheduled_method(f"job_{i}")
        
        # Verify counters
        self.assertEqual(self.scheduler.job_count, 5)
        self.assertEqual(self.scheduler.daily_executions, 5)
        
        # Reset daily count
        self.scheduler.reset_daily_count()
        
        # Verify only daily count was reset
        self.assertEqual(self.scheduler.job_count, 5)
        self.assertEqual(self.scheduler.daily_executions, 0)
        
        # Execute more jobs
        for i in range(3):
            self.scheduler.your_scheduled_method(f"new_job_{i}")
        
        # Verify new counts
        self.assertEqual(self.scheduler.job_count, 8)
        self.assertEqual(self.scheduler.daily_executions, 3)


class TestEdgeCasesAndErrorConditions(unittest.TestCase):
    """Test cases for edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scheduler.signal.signal'):
            self.scheduler = CronScheduler()
    
    def test_extreme_valid_cron_values(self):
        """Test extreme but valid cron expression values."""
        extreme_cases = [
            "0 0 1 1 0",      # New Year's Day if it's Sunday
            "59 23 31 12 6",  # Last minute of year if Dec 31 is Saturday
            "0 0 29 2 *",     # Feb 29 (leap year consideration)
            "*/1 * * * *",    # Every minute (very frequent)
        ]
        
        for cron_expr in extreme_cases:
            with self.subTest(cron_expr=cron_expr):
                job_name = f"extreme_{cron_expr.replace(' ', '_').replace('*', 'star').replace('/', 'slash')}"
                
                # Should not raise an exception
                try:
                    self.scheduler.add_cron_job(cron_expr, job_name)
                    # Clean up
                    self.scheduler.remove_job(job_name)
                except Exception as e:
                    self.fail(f"Valid cron expression {cron_expr} raised exception: {e}")
    
    def test_scheduler_with_no_jobs(self):
        """Test scheduler behavior with no jobs scheduled."""
        # Should not raise exceptions
        with patch('builtins.print'):
            self.scheduler.list_jobs()
        
        # Should handle empty job list gracefully
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 0)
    
    def test_multiple_operations_sequence(self):
        """Test sequence of multiple operations."""
        # Add jobs
        self.scheduler.add_cron_job("0 9 * * *", "job1")
        self.scheduler.add_cron_job("0 17 * * *", "job2")
        
        # List jobs
        with patch('builtins.print'):
            self.scheduler.list_jobs()
        
        # Remove one job
        self.scheduler.remove_job("job1")
        
        # Add another job
        self.scheduler.add_cron_job("0 12 * * *", "job3")
        
        # Verify final state
        jobs = self.scheduler.scheduler.get_jobs()
        self.assertEqual(len(jobs), 2)
        
        job_ids = [job.id for job in jobs]
        self.assertIn("job2", job_ids)
        self.assertIn("job3", job_ids)
        self.assertNotIn("job1", job_ids)


# Test runner and utilities
def run_all_tests(verbosity=2):
    """Run all test cases with specified verbosity."""
    # Get all test classes
    test_classes = [
        TestCronSchedulerInitialization,
        TestScheduledMethod,
        TestCronJobManagement,
        TestThreeDailyJobsSetup,
        TestJobManagementOperations,
        TestDailyCountOperations,
        TestCustomJobsPlaceholder,
        TestSchedulerLifecycle,
        TestEventListeners,
        TestSignalHandling,
        TestValidCronExpressions,
        TestIntegrationWorkflows,
        TestEdgeCasesAndErrorConditions,
    ]
    
    # Create test suite
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    
    return result


def run_specific_test_class(test_class_name, verbosity=2):
    """Run a specific test class by name."""
    test_classes = {
        'TestCronSchedulerInitialization': TestCronSchedulerInitialization,
        'TestScheduledMethod': TestScheduledMethod,
        'TestCronJobManagement': TestCronJobManagement,
        'TestThreeDailyJobsSetup': TestThreeDailyJobsSetup,
        'TestJobManagementOperations': TestJobManagementOperations,
        'TestDailyCountOperations': TestDailyCountOperations,
        'TestCustomJobsPlaceholder': TestCustomJobsPlaceholder,
        'TestSchedulerLifecycle': TestSchedulerLifecycle,
        'TestEventListeners': TestEventListeners,
        'TestSignalHandling': TestSignalHandling,
        'TestValidCronExpressions': TestValidCronExpressions,
        'TestIntegrationWorkflows': TestIntegrationWorkflows,
        'TestEdgeCasesAndErrorConditions': TestEdgeCasesAndErrorConditions,
    }
    
    if test_class_name not in test_classes:
        print(f"❌ Test class '{test_class_name}' not found!")
        print("Available test classes:")
        for name in test_classes.keys():
            print(f"  - {name}")
        return None
    
    test_class = test_classes[test_class_name]
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def run_specific_test_method(test_class_name, test_method_name, verbosity=2):
    """Run a specific test method."""
    test_classes = {
        'TestCronSchedulerInitialization': TestCronSchedulerInitialization,
        'TestScheduledMethod': TestScheduledMethod,
        'TestCronJobManagement': TestCronJobManagement,
        'TestThreeDailyJobsSetup': TestThreeDailyJobsSetup,
        'TestJobManagementOperations': TestJobManagementOperations,
        'TestDailyCountOperations': TestDailyCountOperations,
        'TestCustomJobsPlaceholder': TestCustomJobsPlaceholder,
        'TestSchedulerLifecycle': TestSchedulerLifecycle,
        'TestEventListeners': TestEventListeners,
        'TestSignalHandling': TestSignalHandling,
        'TestValidCronExpressions': TestValidCronExpressions,
        'TestIntegrationWorkflows': TestIntegrationWorkflows,
        'TestEdgeCasesAndErrorConditions': TestEdgeCasesAndErrorConditions,
    }
    
    if test_class_name not in test_classes:
        print(f"❌ Test class '{test_class_name}' not found!")
        return None
    
    test_class = test_classes[test_class_name]
    suite = unittest.TestSuite()
    suite.addTest(test_class(test_method_name))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def generate_test_report(result):
    """Generate a detailed test report."""
    print("\n" + "="*80)
    print("📊 DETAILED TEST REPORT")
    print("="*80)
    
    print(f"📈 Tests Run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"📊 Success Rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"  {i}. {test}")
            # Extract the assertion error message
            lines = traceback.split('\n')
            assertion_line = next((line for line in lines if 'AssertionError' in line), "")
            if assertion_line:
                print(f"     💬 {assertion_line.strip()}")
            print()
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"  {i}. {test}")
            # Extract the error message
            lines = traceback.split('\n')
            error_line = next((line for line in reversed(lines) if line.strip() and not line.startswith(' ')), "")
            if error_line:
                print(f"     💬 {error_line.strip()}")
            print()
    
    print("="*80)
    
    if not result.failures and not result.errors:
        print("🎉 ALL TESTS PASSED! 🎉")
    else:
        print("⚠️  Some tests need attention. Check the details above.")
    
    print("="*80)


if __name__ == "__main__":
    import sys
    
    print("🧪 CronScheduler Test Suite")
    print("="*50)
    
    # Check if specific test class or method was requested
    if len(sys.argv) > 1:
        if len(sys.argv) == 2:
            # Run specific test class
            test_class_name = sys.argv[1]
            print(f"🎯 Running specific test class: {test_class_name}")
            result = run_specific_test_class(test_class_name)
        elif len(sys.argv) == 3:
            # Run specific test method
            test_class_name = sys.argv[1]
            test_method_name = sys.argv[2]
            print(f"🎯 Running specific test: {test_class_name}.{test_method_name}")
            result = run_specific_test_method(test_class_name, test_method_name)
        else:
            print("❌ Usage: python test_scheduler.py [TestClassName] [test_method_name]")
            sys.exit(1)
    else:
        # Run all tests
        print("🚀 Running all test classes...")
        result = run_all_tests()
    
    if result:
        generate_test_report(result)
        
        # Exit with appropriate code
        if result.failures or result.errors:
            sys.exit(1)
        else:
            sys.exit(0)