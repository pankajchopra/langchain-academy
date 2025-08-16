Here are several high‑quality resources—including PDFs and research papers—that explore **agentic architectures**, both **single-agent with tool use** and **multi-agent systems**. These are carefully selected to be actionable, technical, and encourage a traditional, well‑grounded understanding. Confidence level: **\~90%**.

---

## &#x20;Key Surveys & Architecture Frameworks

### 1. *The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey* (Apr 2024)

This foundational survey reviews both **single-agent** and **multi-agent** architectures, comparing planning styles, tool usage, role definition, coordination protocols, and architectural trade‑offs. ([arXiv][1])

**PDF available**: [PDF version](https://arxiv.org/pdf/2404.11584)

---

### 2. *Distinguishing Autonomous AI Agents from Collaborative Agentic Systems* (Jun 2025)

A crisp academic framework that clearly differentiates **single (autonomous) agents** from **collaborative multi-agent ecosystems**. Covers memory, planning, deployment, decision-making workflows, and use-case categorizations. ([arXiv][2])

---

### 3. *Multi-agent Architecture Search via Agentic Supernet* (ICML 2025)

Presents **MaAS**, a novel architecture that uses a *supernet* to dynamically select sub-architectures per query, optimizing both performance and resource usage across distributed agents. ([OpenReview][3])

PDF is accessible on OpenReview.

---

### 4. *Magentic‑One: A Generalist Multi-Agent System for Solving Complex Tasks* (Nov 2024)

Introduces **Magentic‑One**, an open‑source multi‑agent system with orchestration and specialized subagents. Demonstrates modularity and task delegation across AutoGenBench benchmarks. ([arXiv][4])

---

### 5. *A Survey of Agentic AI, Multi-Agent Systems and Multimodal Frameworks* (Dec 2024)

Broad overview of frameworks like LangChain, RLlib, CrewAI, and multimodal agentic architectures employed in real-world settings such as health, finance, retail. Useful for evaluating tool‑enabled single agents vs. MAS. ([ResearchGate][5])

---

## &#x20;Tool‑Oriented Single‑Agent Architectures

### 6. *A Practical Guide to Building Agents* (OpenAI PDF, May 2025)

Details how to build **single-agent systems** by incrementally adding tools, orchestration loops, guardrails, and dynamic tool chains—all without jumping prematurely into multi-agent complexity. ([OpenAI CDN][6])

---

### 7. *LLM‑Based Agents for Tool Learning: A Survey* (Jan 2025)

Systematic exploration of architectures where agents learn to discover, select, and invoke tools. Discusses both training-based and non-training-based tool-retrieval methods, planning modules, and emerging multimodal tool integration. ([link.springer.com][7])

---

## &#x20;Classic Foundations & Multi‑Agent Deep RL

### 8. *Deep Reinforcement Learning for Multi-Agent Systems: A Review* (2018)

An earlier review of **deep RL** techniques in MAS—addresses non-stationarity, training schemes, continuous action spaces, and applications. A solid baseline for historical comparison. ([arXiv][8])

---

## &#x20;Additional Architecture Analyses & Case Studies

* \**Artificial Intelligence Agents and Agentic Workflows: the New Frontier of Automation* – covers taxonomy (base vs orchestrator agents), memory, ethical automation scales.([ResearchGate][9])
* \**Agentic AI and Multi-Agent Systems* (AI‑Mind article) – practical explanation of single vs multi‑agent in plain English. Great overview for orienting broad design decisions.([AI Mind][10])
* \**Agent design pattern catalogue* – outlines architectural patterns, registries, planning, reflection designs (Andrew Ng’s approach).([sciencedirect.com][11])

---

## &#x20;Summary Table

| Topic                             | Recommended Paper & Focus                    |
| --------------------------------- | -------------------------------------------- |
| **Single‑agent with tools**       | OpenAI guide, LLM‑based tool learning survey |
| **Multi‑agent architectures**     | Magentic‑One, Agentic Supernet, ICML 2025    |
| **Mixed survey and taxonomy**     | Masterman et al. (2024), Ramachandran (2024) |
| **Foundational RL-based MAS**     | Nguyen et al. (2018)                         |
| **Taxonomies & pattern catalogs** | Bansod (2025), Agent design patterns         |

---

Let me know if you want summaries of specific papers, code examples, or implementation guidance on any of these architectures!

[1]: https://arxiv.org/pdf/2404.11584?utm_source=chatgpt.com "[PDF] arXiv:2404.11584v1 [cs.AI] 17 Apr 2024"
[2]: https://arxiv.org/abs/2506.01438?utm_source=chatgpt.com "Distinguishing Autonomous AI Agents from Collaborative Agentic Systems: A Comprehensive Framework for Understanding Modern Intelligent Architectures"
[3]: https://openreview.net/forum?id=imcyVlzpXh&utm_source=chatgpt.com "Multi-agent Architecture Search via Agentic Supernet | OpenReview"
[4]: https://arxiv.org/abs/2411.04468?utm_source=chatgpt.com "Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks"
[5]: https://www.researchgate.net/publication/387577302_A_Survey_of_Agentic_AI_Multi-Agent_Systems_and_Multimodal_Frameworks_Architectures_Applications_and_Future_Directions?utm_source=chatgpt.com "(PDF) A Survey of Agentic AI, Multi-Agent Systems, and Multimodal ..."
[6]: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf?utm_source=chatgpt.com "[PDF] A practical guide to building agents - OpenAI"
[7]: https://link.springer.com/article/10.1007/s41019-025-00296-9?utm_source=chatgpt.com "LLM-Based Agents for Tool Learning: A Survey"
[8]: https://arxiv.org/abs/1812.11794?utm_source=chatgpt.com "Deep Reinforcement Learning for Multi-Agent Systems: A Review of Challenges, Solutions and Applications"
[9]: https://www.researchgate.net/publication/389738694_Artificial_intelligence_agents_and_agentic_workflows_the_new_frontier_of_automation?utm_source=chatgpt.com "(PDF) Artificial intelligence agents and agentic workflows: the new ..."
[10]: https://pub.aimind.so/agentic-ai-and-multi-agent-systems-cc32803e1f52?utm_source=chatgpt.com "Agentic AI and Multi-Agent Systems | by Abduldattijo - AI Mind"
[11]: https://www.sciencedirect.com/science/article/pii/S0164121224003224?utm_source=chatgpt.com "Agent design pattern catalogue: A collection of architectural patterns ..."agentic_