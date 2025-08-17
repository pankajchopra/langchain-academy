%%{init: {'theme':'base', 'themeVariables': {'primaryColor': '#e1f5fe', 'primaryTextColor': '#0277bd', 'primaryBorderColor': '#0277bd', 'lineColor': '#ef6c00', 'sectionBkColor': '#f3e5f5', 'altSectionBkColor': '#fff3e0', 'gridColor': '#ccc', 'secondaryColor': '#e8f5e8', 'tertiaryColor': '#ffebee'}}}%%

gantt
    title Async vs Synchronous Execution Performance Comparison
    dateFormat X
    axisFormat %L ms
    
    section 🐌 Synchronous Execution
    Fibonacci Tool (Local)           :sync1, 0, 180
    Wait for Fibonacci               :milestone, 180
    Math Tool (MCP Stdio)           :sync2, 180, 400  
    Wait for Math                   :milestone, 400
    Weather Tool (MCP HTTP)         :sync3, 400, 750
    Wait for Weather                :milestone, 750
    Total Synchronous Time          :crit, sync-total, 0, 750
    
    section ⚡ Async Concurrent Execution  
    Fibonacci Tool (Async)          :async1, 0, 180
    Math Tool (Async)               :async2, 0, 220
    Weather Tool (Async)            :async3, 0, 350
    All Tools Complete              :milestone, 350
    Total Async Time                :done, async-total, 0, 350
    Performance Improvement         :active, improvement, 350, 750
    
    section 📊 Resource Utilization
    CPU Usage (Sync)                :resource1, 0, 750
    Memory Usage (Sync)             :resource2, 0, 750
    Network I/O (Sync)              :resource3, 180, 750
    CPU Usage (Async)               :done, resource4, 0, 350
    Memory Usage (Async)            :done, resource5, 0, 350  
    Network I/O (Async)             :done, resource6, 0, 350
    
    section 🎯 Key Metrics
    Latency Reduction               :metric1, 350, 750
    Throughput Increase             :done, metric2, 0, 350
    Resource Efficiency             :done, metric3, 0, 350
    Error Recovery Time             :done, metric4, 0, 100