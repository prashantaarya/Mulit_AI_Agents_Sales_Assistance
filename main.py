"""
Main entry point - orchestrates the entire system
"""
import os
from functools import partial
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import Tool, tool
from typing import Optional
import json

from data.processor import process_data
from tools.toolbox import ToolBox
from tools.hybrid_search import HybridSearchToolBox
from agents.prospecting import create_prospecting_agent, prospecting_node
from agents.insights import create_insights_agent, insights_node
from agents.communication import create_communication_agent, communication_node
from graph.router import create_router_chain, route_requests
from graph.workflow import create_workflow
from utils.helpers import run_conversation


class SalesSystem:
    """Main Sales System orchestrator."""
    
    def __init__(self):
        self.llm = None
        self.app = None
        self.enhanced_toolbox = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the complete sales system."""
        print("Initializing Sales System...")
        
        # Load environment variables
        load_dotenv()
        
        # Get and clean environment variables
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip('"').strip("'")
        model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-120b").strip('"').strip("'")
        temperature = float(os.getenv("TEMPERATURE", "0.1").strip('"').strip("'"))
        data_file_path = os.getenv("DATA_FILE_PATH", "data/prospects.xlsx").strip('"').strip("'")
        
        # Validate API key
        if not groq_api_key:
            print("ERROR: GROQ_API_KEY not found in .env file")
            return
            
        # Validate data file
        if not os.path.exists(data_file_path):
            print(f"ERROR: Data file not found at: {data_file_path}")
            return
        
        print(f"API Key loaded: {groq_api_key[:10]}...")
        
        # Initialize LLM - Set both environment and explicit key
        os.environ["GROQ_API_KEY"] = groq_api_key
        try:
            self.llm = ChatGroq(
                temperature=temperature,
                model_name=model_name,
                api_key=groq_api_key  # Use api_key parameter instead of groq_api_key
            )
            print("LLM initialized successfully.")
        except Exception as e:
            print(f"LLM initialization failed: {e}")
            return
        
        # Test LLM immediately
        try:
            test_response = self.llm.invoke("Hello")
            print(f"LLM test successful: {test_response.content[:50]}...")
        except Exception as e:
            print(f"LLM test failed: {e}")
            return
        
        # Process data
        processed_df = process_data(data_file_path)
        if processed_df.empty:
            print("Data processing failed")
            return
        
        # Initialize toolboxes
        try:
            toolbox = ToolBox(dataframe=processed_df)
            self.enhanced_toolbox = HybridSearchToolBox(df=processed_df, llm=self.llm)
            print("Tools configured successfully.")
        except Exception as e:
            print(f"Toolbox initialization failed: {e}")
            return
        
        # Create tools with proper scope binding
        def create_tools():
            @tool
            def find_prospects_hybrid(query: str) -> str:
                """
                Find prospects using intelligent search.
                Args:
                    query: Search query string
                Returns:
                    JSON string of results
                """
                try:
                    if not query or not isinstance(query, str):
                        return json.dumps({"error": "Invalid query parameter"})
                    
                    clean_query = str(query).strip()
                    if not clean_query:
                        return json.dumps({"error": "Empty query provided"})
                    
                    result = self.enhanced_toolbox.find_prospects_hybrid(clean_query)
                    
                    if not isinstance(result, (list, dict)):
                        result = {"error": "Invalid result format", "message": str(result)}
                    
                    return json.dumps(result, ensure_ascii=False, default=str)
                    
                except Exception as e:
                    return json.dumps({
                        "error": f"Search failed: {str(e)[:100]}",
                        "suggestion": "Try a simpler query"
                    })

            @tool 
            def get_prospect_details(prospect_name: str) -> str:
                """
                Get detailed analysis for a specific prospect.
                Args:
                    prospect_name: Name of the business to analyze
                Returns:
                    JSON string of prospect details
                """
                try:
                    if not prospect_name or not isinstance(prospect_name, str):
                        return json.dumps({"error": "Invalid prospect name parameter"})
                    
                    clean_name = str(prospect_name).strip()
                    if not clean_name:
                        return json.dumps({"error": "Empty prospect name provided"})
                    
                    result = self.enhanced_toolbox.get_prospect_details(clean_name)
                    
                    if result is None:
                        return json.dumps({
                            "error": "Prospect not found",
                            "suggestion": f"No business found with name: {clean_name}"
                        })
                    
                    if not isinstance(result, dict):
                        result = {"error": "Invalid result format", "data": str(result)}
                    
                    return json.dumps(result, ensure_ascii=False, default=str)
                    
                except Exception as e:
                    return json.dumps({
                        "error": f"Analysis failed: {str(e)[:100]}",
                        "suggestion": "Check business name spelling"
                    })
            
            return find_prospects_hybrid, get_prospect_details
        
        # Get tool instances
        find_prospects_hybrid, get_prospect_details = create_tools()
        
        # Create agents
        prospecting_tools = [find_prospects_hybrid]
        insights_tools = [get_prospect_details]
        communication_tools = [get_prospect_details] 
        
        try:
            prospecting_agent = create_prospecting_agent(self.llm, prospecting_tools)
            insights_agent = create_insights_agent(self.llm, insights_tools)
            communication_agent = create_communication_agent(self.llm, communication_tools)
            print("Agents created successfully.")
        except Exception as e:
            print(f"Agent creation failed: {e}")
            return
        
        # Create router
        try:
            router_chain = create_router_chain(self.llm)
            print("Router created successfully.")
        except Exception as e:
            print(f"Router creation failed: {e}")
            return
        
        # Create workflow
        def prospecting_node_func(state):
            return prospecting_node(state, prospecting_agent)
        
        def insights_node_func(state):
            return insights_node(state, insights_agent)
        
        def communication_node_func(state):
            return communication_node(state, communication_agent)
        
        def route_requests_func(state):
            return route_requests(state, router_chain)
        
        try:
            self.app = create_workflow(
                prospecting_node_func, 
                insights_node_func, 
                communication_node_func, 
                route_requests_func
            )
            print("Graph compiled successfully! System ready.")
        except Exception as e:
            print(f"Workflow creation failed: {e}")
            return
    
    def run_query(self, query: str):
        """Run a query through the sales system."""
        if self.app is None:
            print("System not initialized properly.")
            return None
        return run_conversation(self.app, query)
    
    def get_system_status(self):
        """Get the current system status."""
        return {
            "LLM": "Ready" if self.llm else "Not initialized",
            "App": "Ready" if self.app else "Not initialized", 
            "Enhanced Toolbox": "Ready" if self.enhanced_toolbox else "Not initialized"
        }


def main():
    """Main function to run the sales system."""
    print("Initializing AI Sales System...")
    sales_system = SalesSystem()
    
    status = sales_system.get_system_status()
    
    if not all("Ready" in s for s in status.values()):
        print("\nSYSTEM INITIALIZATION FAILED")
        print("System Status:")
        for component, status_msg in status.items():
            print(f"   {component}: {status_msg}")
        return None
    
    print("System ready!")
    
    while True:
        try:
            user_input = input("\nEnter your query (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter a query.")
                continue
            
            print(f"Processing: '{user_input}'")
            
            try:
                result = sales_system.run_query(user_input)
                
                if result and 'agent_out' in result:
                    response = result['agent_out']['messages'][0].content
                    print("\nRESULT:")
                    print("-" * 40)
                    print(response)
                    print("-" * 40)
                else:
                    print("No result returned.")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
    
    return sales_system


if __name__ == "__main__":
    system = main()