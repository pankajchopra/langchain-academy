```mermaid

graph TD
    subgraph FinancialAdvisorAgent [Main Financial Advisor Agent]
        direction LR
        A[User Query] --> B(agent: LLM Brain);
        B -- "Needs info/delegate?" --> C[custom_tool_node];
        
        subgraph A2A_Client_Logic [Inside custom_tool_node]
            direction TB
            C1{Tool Call?}
            C1 -- "get_client_profile" --> C2[Execute get_client_profile];
            C1 -- "submit_form_filling_task" --> C3[1. POST /submit_task];
            C1 -- "check_task_status" --> C4[2. GET /task_status/task_id];
        end

        C --> D[Update State];
        D --> B;

        B -- "Ready to answer?" --> E{responsible_ai_filter};
        E -- "Validation Fails" --> B;
        E -- "Validation Passes" --> F([Final Answer]);
    end

    subgraph PDF-Form-Filler-Agent [Specialized Agent: PDF Form Filler -Async]
        direction LR
        G[A2A Endpoints]
        subgraph API_Endpoints [FastAPI Server]
            direction TB
            G1["/submit_task (POST)"];
            G2["/task_status/{task_id} (GET)"];
        end
        G --> H{Task Storage};
        H -- "Run in Background" --> I(Internal LangGraph Agent);
    end
    
    subgraph ClientContextAgent_Placeholder [Future: ClientContextAgent]
        style ClientContextAgent_Placeholder fill:#f0f0f0,stroke:#999,stroke-dasharray: 5 5
        J(A2A Endpoint);
    end

    %% A2A Communication Links
    C3 -.-> G1;
    C4 -.-> G2;
    G1 -.-> C3;
    G2 -.-> C4;
    
    %% Styling
    style FinancialAdvisorAgent fill:#e6f3,stroke:#333,stroke-width:2px
    style PDF-Form-Filler-Agent fill:#d4edda,stroke:#155724,stroke-width:2px
    style F fill:#f8d7da,stroke:#721c24,stroke-width:2px

```

    How to Read This Diagram
Main Agent's Logic: The FinancialAdvisorAgent now contains a custom_tool_node which acts as the A2A client. It decides whether to call a simple tool like get_client_profile or to start the multi-step A2A process.

Asynchronous A2A Flow: The numbered arrows clearly show the new asynchronous pattern:

The main agent first makes a POST request to /submit_task.

It then makes one or more GET requests to /task_status to poll for the result.

PDF Agent's Role: The PDF-Form-Filler-Agent is shown as a server with distinct API endpoints. It manages tasks in a Task Storage and processes them in the background using its own internal agent.

Future Work: The ClientContextAgent is included as a grayed-out, dashed box to clearly indicate that it's the next component we plan to build and integrate.