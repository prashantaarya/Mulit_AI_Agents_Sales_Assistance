# agents/base.py - Updated with better error handling
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from graph.state import GraphState

def create_agent(llm, tools: list, system_prompt: str):
    """Helper function to create a new agent with robust error handling."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), 
        MessagesPlaceholder(variable_name="messages"), 
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    executor = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=False,  # Reduce verbosity
        return_intermediate_steps=False,  # Reduce output
        max_iterations=3,  # Limit iterations
        max_execution_time=20,  # 20 second timeout
        early_stopping_method="generate",
        handle_parsing_errors=True  # Handle parsing errors gracefully
    )
    return executor

def agent_node(state: GraphState, agent: AgentExecutor, name: str):
    """Helper function to invoke an agent with error handling."""
    try:
        result = agent.invoke(state)
        output = result.get("output", "No output generated")
        
        # Ensure output is a string
        if not isinstance(output, str):
            output = str(output)
            
        return {"messages": [AIMessage(content=output, name=name)]}
        
    except Exception as e:
        error_msg = f"Agent {name} encountered an error: {str(e)[:100]}. Please try a different approach."
        return {"messages": [AIMessage(content=error_msg, name=name)]}