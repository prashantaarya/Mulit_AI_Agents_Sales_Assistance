AI-Driven Multi-Agent System for Sales Enablement-
This repository contains a sophisticated, multi-agent conversational AI system designed to act as a co-pilot for sales and demand generation professionals. It leverages a powerful open-source Large Language Model (LLM) through Groq's LPU Inference Engine to provide intelligent, personalized, and context-aware insights on SMB prospects.

The system is built on a philosophy of efficiency and precision, with an architecture designed to break down complex queries into manageable tasks for specialized AI agents.

Features-    
Multi-Agent Workflow: Utilizes LangGraph to create a robust workflow where a primary Router Agent delegates tasks to specialized agents for Prospecting, Insights, and Communication.

Role-Based Personalization: The Router Agent intelligently identifies the user's role (Sales Rep vs. Demand Gen) and tailors the conversation and output accordingly.

Deep Prospect Intelligence: Goes beyond simple data retrieval to provide rich analysis, including SWOT, digital presence scores, key business gaps, and strategic opportunities.

Conversational Memory: Maintains context across multiple turns in a conversation for a coherent and natural user experience.

Interactive UI: A user-friendly web interface built with Streamlit allows for easy interaction, chat history management, and clear visualization of results.

Efficient LLM Usage: The architecture is designed to make efficient, targeted calls to the LLM, making it well-suited for high-performance open-source models.

System Architecture-
The system is designed with a modular and scalable architecture that ensures a clear separation of concerns.

Frontend (app.py): A Streamlit application serves as the user interface, capturing user queries and displaying formatted results.

Orchestrator (main.py): The SalesSystem class initializes all components—the LLM, data, tools, and agents—and compiles the final LangGraph workflow.

LangGraph Workflow (graph/):

Router: The entry point to the graph. A lightweight LLM call analyzes the user's query and conversation history to determine intent and user role.

State: A shared state object passes messages, user context, and data between the nodes in the graph.

Agents: Specialized worker agents (ProspectingAgent, InsightsAgent, CommunicationAgent) execute the delegated tasks.

Intelligence Layer (tools/): The HybridSearchToolBox is the analytical engine. It uses a focused LLM chain to translate natural language into precise data filters, which are then used by Python to perform analysis.

Data Layer (data/): Prospect data is loaded from an Excel file and processed into a clean, queryable Pandas DataFrame.

Technology Stack -
Backend: Python 3.10+

AI Frameworks: LangChain, LangGraph

LLM Provider: Groq (using the deepseek-r1-distill-llama-70b model)

Data Handling: Pandas

Frontend: Streamlit

Vector Store: FAISS (for future semantic search capability)

Embeddings: Sentence-Transformers

Setup and Installation
Follow these steps to get the system up and running locally.

1. Clone the Repository
Bash

git clone <repository-url>
cd <repository-directory>
2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install Dependencies
Install all required packages from the requirements.txt file.

Bash

pip install -r requirements.txt
(Note: A requirements.txt file should be created containing packages like langchain, langgraph, langchain-groq, streamlit, pandas, openpyxl, sentence-transformers, faiss-cpu.)

4. Configure Environment Variables
The system requires an API key from Groq. You should never hardcode keys in the code.
Create a .env file in the root of the project and add your API key:

GROQ_API_KEY="your-groq-api-key-here"
The application will load this key from the environment.

5. Place Data File
Ensure your prospect data file, Sample Data for the Model.xlsx, is located in the ./data/ directory.

How to Run
The application can be run in two modes: via the interactive Streamlit UI or the command-line interface.

1. Interactive Web UI (Recommended)
To start the Streamlit application, run the following command in your terminal:

Bash

streamlit run app.py
Open your web browser and navigate to the local URL provided (usually http://localhost:8501).

2. Command-Line Interface
For quick tests or running in a terminal-only environment, you can use main.py:

Bash

python main.py
This will present you with a menu to enter an interactive CLI mode, run predefined test queries, or process a single query.

Example Queries
You can try the following queries in either interface:

Find businesses in the Computer Contractors category with low local presence but high google ads spend.

What are the strengths and weaknesses of ABC Plumbing?

Draft a personalized email for a xyz business and prose my product based on their needs

Find businesses in the Computer Contractors category in Texas, with low local presence but high google ads spend.

Strategic Design Choices-
This project includes several deliberate design choices to ensure a robust, efficient, and extensible system.

Prioritizing Precision for Core Functionality
The current search mechanism in HybridSearchToolBox uses an LLM to extract structured filters from a query, which are then applied to the DataFrame. This approach was chosen as the foundational search method because it guarantees high precision and control. For a business application, it is critical that specific queries (e.g., "find contractors in Texas") return exact and verifiable results, which this method provides. This also ensures that the initial LLM calls are lightweight and fast, aligning with a resource-conscious design philosophy.

Foundational Work for Future Enhancements
The codebase demonstrates foresight by including components that are part of a planned, phased development roadmap.

Semantic Search Readiness (FAISS): The system currently builds a FAISS vector store during initialization but does not use it for querying. This is intentional. Building the vector index is a one-time setup cost. By having it ready, the system is primed for Phase 2: Activating Semantic Search. This future enhancement will allow for broader, more conceptual queries and can be implemented with minimal changes to the existing architecture.

Architectural Evolution (toolbox.py): The presence of toolbox.py alongside the more advanced HybridSearchToolBox showcases the project's iterative development. toolbox.py represents a successful initial prototype. The decision to use HybridSearchToolBox exclusively for the final agent tools demonstrates a commitment to leveraging the most powerful and sophisticated implementation developed during the project's lifecycle.

Future Work & Roadmap

Activate Semantic Search: The highest-priority enhancement is to activate the pre-built FAISS vector store to enable true hybrid search, combining the precision of filtering with the power of conceptual search.

Harden Security: Remove all hardcoded API keys from the source code and ensure they are loaded exclusively from environment variables or a secure secret management service.

Refactor Memory Management: For multi-user scalability, transition the conversational memory from a global dictionary to a session-scoped object or a dedicated cache like Redis.

Live CRM Integration: Extend the agent's tools to connect to live CRM platforms (e.g., Salesforce, HubSpot) to fetch real-time data and update prospect records.

Formal Testing Suite: Develop a comprehensive suite of unit and integration tests to ensure the system's long-term reliability and stability.