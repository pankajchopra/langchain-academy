#!/usr/bin/env python3
"""
Cron Expression Daily Scheduler Template
A Python application that triggers a method three times a day using cron expressions.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import logging
import time
import signal
import sys
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cron_scheduler.log'),
        logging.StreamHandler()
    ]
)

class CronScheduler:
    """A scheduler class that runs methods using cron expressions."""
    
    def __init__(self, use_background: bool = False):
        """
        Initialize the scheduler and set up event listeners and signal handlers.
        
        Args:
            use_background (bool): If True, uses BackgroundScheduler, 
                                 otherwise uses BlockingScheduler
        """
        self.job_count = 0
        self.daily_executions = 0
        self.use_background = use_background
        
        if use_background:
            self.scheduler = BackgroundScheduler()
        else:
            self.scheduler = BlockingScheduler()
        
        # Add event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def your_scheduled_method(self, job_name: str = "default"):
        """
        The main scheduled task to be executed by the scheduler.
        Replace with your business logic.
        
        Args:
            job_name (str): Name identifier for the job
        """
        self.job_count += 1
        self.daily_executions += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"🚀 Scheduled task '{job_name}' executed at {current_time}")
        print(f"📊 Total executions: {self.job_count} | Today: {self.daily_executions}")
        
        # Add your actual business logic here
        try:
            # Example: Your actual work goes here
            print(f"   ✅ Performing scheduled task: {job_name}")
            
            # Simulate some work
            time.sleep(1)
            
            print(f"   ✅ Task '{job_name}' completed successfully!")
            logging.info(f"Scheduled task '{job_name}' completed successfully. Total executions: {self.job_count}")
            
        except Exception as e:
            print(f"   ❌ Error in scheduled task '{job_name}': {e}")
            logging.error(f"Error in scheduled task '{job_name}': {e}")
            raise
    
    def add_cron_job(self, cron_expression: str, job_name: str = "cron_job", 
                     method=None, **kwargs):
        """
        Add a job to the scheduler using a cron expression.
        
        Args:
            cron_expression (str): Cron expression (e.g., "0 8,14,20 * * *")
            job_name (str): Unique name for the job
            method: Method to execute (defaults to your_scheduled_method)
            **kwargs: Additional arguments to pass to the method
        
        Cron expression format: "minute hour day month day_of_week"
        Examples:
            "0 8,14,20 * * *"  - Run at 8:00, 14:00, 20:00 every day
            "30 */4 * * *"     - Run every 4 hours at 30 minutes past
            "0 6,12,18 * * 1-5" - Run at 6:00, 12:00, 18:00 on weekdays
        """
        if method is None:
            method = lambda: self.your_scheduled_method(job_name)
        
        try:
            # Parse and validate cron expression
            trigger = CronTrigger.from_crontab(cron_expression)
            
            # Add the job
            self.scheduler.add_job(
                func=method,
                trigger=trigger,
                id=job_name,
                name=job_name,
                **kwargs
            )
            
            print(f"📅 Added cron job '{job_name}' with expression: {cron_expression}")
            logging.info(f"Added cron job '{job_name}' with expression: {cron_expression}")
            
        except Exception as e:
            print(f"❌ Error adding cron job '{job_name}': {e}")
            logging.error(f"Error adding cron job '{job_name}': {e}")
            raise
    
    def setup_three_daily_jobs(self, times: List[int] = None):
        """
        Configure the scheduler to run a job three times daily at specified hours.
        Also adds a daily reset job.
        
        Args:
            times (List[int]): List of hours (0-23). Defaults to [8, 14, 20]
        """
        if times is None:
            times = [8, 14, 20]  # 8 AM, 2 PM, 8 PM
        
        if len(times) != 3:
            raise ValueError("Exactly 3 times (hours) must be provided")
        
        # Validate hours
        for hour in times:
            if not 0 <= hour <= 23:
                raise ValueError(f"Hour {hour} must be between 0 and 23")
        
        # Create cron expression for the three times
        hours_str = ",".join(map(str, times))
        cron_expression = f"0 {hours_str} * * *"
        
        # Add the job
        self.add_cron_job(cron_expression, "three_daily_executions")
        
        # Add daily counter reset job at midnight
        self.add_cron_job("1 0 * * *", "daily_reset", self.reset_daily_count)
        
        print(f"🎯 Configured for three daily executions at hours: {times}")
        return cron_expression
    
    def reset_daily_count(self):
        """
        Reset the daily execution counter to zero.
        """
        self.daily_executions = 0
        print("🔄 Daily execution counter reset")
        logging.info("Daily execution counter reset")
    
    def add_custom_jobs(self):
        """
        Add custom jobs to the scheduler.
        Modify this function to add your own cron jobs.
        """
        # Example custom jobs:
        
        # Every hour during business hours (9 AM to 5 PM, weekdays)
        # self.add_cron_job("0 9-17 * * 1-5", "business_hours", 
        #                   lambda: self.your_scheduled_method("business_hours"))
        
        # Every 15 minutes during specific hours
        # self.add_cron_job("*/15 8-20 * * *", "frequent_check",
        #                   lambda: self.your_scheduled_method("frequent_check"))
        
        # Weekly report on Mondays at 9 AM
        # self.add_cron_job("0 9 * * 1", "weekly_report",
        #                   lambda: self.your_scheduled_method("weekly_report"))
        
        # Monthly report on first day of month at 10 AM
        # self.add_cron_job("0 10 1 * *", "monthly_report",
        #                   lambda: self.your_scheduled_method("monthly_report"))
        
        pass
    
    def list_jobs(self):
        """
        Print all scheduled jobs and their next run times.
        """
        jobs = self.scheduler.get_jobs()
        if jobs:
            print("📋 Current scheduled jobs:")
            for job in jobs:
                next_run = job.next_run_time
                next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S") if next_run else "No next run"
                print(f"   🔹 {job.name} (ID: {job.id})")
                print(f"      Next run: {next_run_str}")
                print(f"      Trigger: {job.trigger}")
        else:
            print("📋 No jobs currently scheduled")
    
    def remove_job(self, job_id: str):
        """
        Remove a job from the scheduler by its ID.
        """
        try:
            self.scheduler.remove_job(job_id)
            print(f"🗑️  Removed job: {job_id}")
            logging.info(f"Removed job: {job_id}")
        except Exception as e:
            print(f"❌ Error removing job {job_id}: {e}")
            logging.error(f"Error removing job {job_id}: {e}")
    
    def start_scheduler(self):
        """
        Start the scheduler and keep it running.
        Handles graceful shutdown on interruption.
        """
        try:
            print("🎯 Cron Scheduler started! Press Ctrl+C to stop.")
            print("📊 Scheduler mode:", "Background" if self.use_background else "Blocking")
            
            self.list_jobs()
            
            if not self.scheduler.running:
                self.scheduler.start()
                
                # If using background scheduler, keep main thread alive
                if self.use_background:
                    try:
                        while True:
                            time.sleep(60)
                            # You can add other main thread work here
                    except KeyboardInterrupt:
                        pass
            
        except KeyboardInterrupt:
            print("\n⏹️  Scheduler stopped by user")
        except Exception as e:
            print(f"❌ Scheduler error: {e}")
            logging.error(f"Scheduler error: {e}")
        finally:
            self.stop_scheduler()
    
    def stop_scheduler(self):
        """
        Stop the scheduler gracefully.
        """
        if self.scheduler.running:
            self.scheduler.shutdown()
            print("⏹️  Scheduler stopped gracefully")
            logging.info("Scheduler stopped gracefully")
    
    def _job_executed(self, event):
        """
        Event listener for successful job execution.
        """
        logging.info(f"Job '{event.job_id}' executed successfully")
    
    def _job_error(self, event):
        """
        Event listener for job execution errors.
        """
        logging.error(f"Job '{event.job_id}' failed: {event.exception}")
        print(f"❌ Job '{event.job_id}' failed: {event.exception}")
    
    def _signal_handler(self, signum, frame):
        """
        Handle system signals for graceful shutdown.
        """
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop_scheduler()
        sys.exit(0)


def main():
    """
    Main entry point to demonstrate the cron scheduler.
    Sets up jobs and starts the scheduler.
    """
    
    # Create scheduler instance (blocking mode)
    scheduler = CronScheduler(use_background=False)
    
    # Option 1: Use default times (8 AM, 2 PM, 8 PM)
    cron_expr = scheduler.setup_three_daily_jobs()
    print(f"📅 Using cron expression: {cron_expr}")
    
    # Option 2: Use custom times (uncomment to use)
    # custom_times = [9, 15, 21]  # 9 AM, 3 PM, 9 PM
    # cron_expr = scheduler.setup_three_daily_jobs(custom_times)
    
    # Option 3: Add individual cron jobs (uncomment to use)
    # scheduler.add_cron_job("0 */6 * * *", "every_6_hours")  # Every 6 hours
    # scheduler.add_cron_job("30 8,12,16,20 * * *", "custom_times")  # Custom times
    
    # Add any custom jobs
    scheduler.add_custom_jobs()
    
    # Start the scheduler
    scheduler.start_scheduler()


def background_example():
    """
    Example of running the scheduler in background mode.
    """
    
    print("🔄 Starting background scheduler example...")
    
    # Create background scheduler
    scheduler = CronScheduler(use_background=True)
    
    # Setup jobs
    scheduler.setup_three_daily_jobs([6, 12, 18])  # 6 AM, 12 PM, 6 PM
    
    # Add a frequent job for demonstration
    scheduler.add_cron_job("*/2 * * * *", "every_2_minutes", 
                          lambda: print(f"⏰ Background job at {datetime.now().strftime('%H:%M:%S')}"))
    
    # Start scheduler
    scheduler.start_scheduler()


def test_cron_expressions():
    """
    Test various cron expressions with short intervals for demonstration.
    """
    
    print("🧪 Testing cron expressions...")
    
    scheduler = CronScheduler(use_background=False)
    
    # Test jobs with short intervals (for testing purposes)
    test_jobs = [
        ("*/1 * * * *", "every_minute", "Runs every minute"),
        ("*/2 * * * *", "every_2_minutes", "Runs every 2 minutes"),
        ("0,30 * * * *", "half_hourly", "Runs at 0 and 30 minutes past each hour"),
    ]
    
    for cron_expr, job_name, description in test_jobs:
        print(f"📝 Adding test job: {description}")
        scheduler.add_cron_job(cron_expr, job_name)
    
    print("\n🧪 Test scheduler will run for 5 minutes...")
    scheduler.start_scheduler()


if __name__ == "__main__":
    # Choose which mode to run:
    
    # 1. Normal scheduler with three daily executions
    main()
    
    # 2. Background scheduler example (uncomment to use)
    # background_example()
    
    # 3. Test mode with short intervals (uncomment to use)
    # test_cron_expressions()


# Installation requirements and cron expression guide:
"""
Installation:
pip install APScheduler

Cron Expression Format:
"minute hour day month day_of_week"

Fields:
- minute: 0-59
- hour: 0-23
- day: 1-31
- month: 1-12
- day_of_week: 0-6 (0 = Sunday)

Special Characters:
- * : Any value
- , : List separator (e.g., 1,3,5)
- - : Range (e.g., 1-5)
- / : Step values (e.g., */5 = every 5)

Common Examples:
"0 8,14,20 * * *"     - 8 AM, 2 PM, 8 PM daily
"*/15 * * * *"        - Every 15 minutes
"0 9-17 * * 1-5"      - Every hour, 9 AM-5 PM, weekdays
"0 0 1 * *"           - First day of every month at midnight
"0 6 * * 1"           - Every Monday at 6 AM
"30 */4 * * *"        - Every 4 hours at 半 past the hour

Features:
- ✅ Standard cron expression syntax
- ✅ Multiple job support
- ✅ Background and blocking modes
- ✅ Event listeners for monitoring
- ✅ Graceful shutdown handling
- ✅ Comprehensive logging
- ✅ Job management (add/remove/list)
- ✅ Error handling and recovery
"""