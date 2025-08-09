# utils/helpers.py - Optimized version
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate   
from langchain_core.messages import SystemMessage
import os
from langchain_groq import ChatGroq

# Simplified memory storage
conversation_memory = {"history": [], "user_context": {}}

def run_conversation(app, query: str):
    """Optimized conversation runner."""
    # Add to memory (keep last 5 only)
    conversation_memory["history"].append({
        "query": query,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    })
    
    if len(conversation_memory["history"]) > 5:
        conversation_memory["history"] = conversation_memory["history"][-5:]
    
    # Simplified context
    context_msg = f"Query: {query}"
    
    initial_state = {
        "messages": [HumanMessage(content=context_msg)],
        "conversation_history": conversation_memory["history"][-2:],  # Only last 2
        "user_context": conversation_memory["user_context"]
    }
    
    output_data = {}

    if app is not None:
        for event in app.stream(initial_state, {"recursion_limit": 3}):  # Reduced limit
            for key, value in event.items():
                print(f"--- {key} ---")
                print(value)
                print("\n" + "="*30 + "\n")
                output_data["agent_out"] = value
                
                # Store minimal response
                if "agent_out" in output_data and value.get("messages"):
                    conversation_memory["history"][-1]["response"] = str(value["messages"][-1].content)[:200]  # Truncate
    else:
        raise RuntimeError("App not compiled. Please fix errors above.")

    return output_data

def get_conversation_context():
    """Get current conversation context."""
    return conversation_memory

def clear_conversation_memory():
    """Clear conversation memory."""
    global conversation_memory
    conversation_memory = {"history": [], "user_context": {}}