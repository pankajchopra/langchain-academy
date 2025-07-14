```mermaid
graph TD
    Agent["agent"]
    Tools["tools"]
    RAI["responsible_ai_filter"]
    End["END"]

    Agent -- |use_tools| --> Tools
    Agent -- validate_output --> RAI
    Tools --> Agent
    RAI --> End

```