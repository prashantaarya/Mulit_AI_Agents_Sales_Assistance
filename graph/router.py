# graph/router.py
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from graph.state import GraphState

def create_router_chain(llm):
    """Create router chain with robust parsing."""
    router_prompt = PromptTemplate(
        template="""
        Route the user request to the appropriate agent.
        
        ROUTING RULES:
        - 'prospecting': Finding/searching businesses, lists, filtering
        - 'insights': Analysis of specific companies, detailed reports  
        - 'communication': Drafting messages/emails, outreach
        - 'end': Goodbye/thank you/exit
        
        User message: {last_message}
        
        Respond with only ONE word: prospecting, insights, communication, or end
        """,
        input_variables=["last_message"],
    )
    
    return router_prompt | llm | StrOutputParser()

def route_requests(state: GraphState, router_chain) -> str:
    """Routes requests with robust error handling."""
    try:
        last_message = state['messages'][-1].content
        result = router_chain.invoke({"last_message": last_message})
        
        # Clean and normalize result
        route = result.strip().lower()
        
        # Map variations to standard routes
        route_mappings = {
            'prospect': 'prospecting',
            'search': 'prospecting', 
            'find': 'prospecting',
            'list': 'prospecting',
            'insight': 'insights',
            'analysis': 'insights',
            'analyze': 'insights',
            'detail': 'insights',
            'communicate': 'communication',
            'email': 'communication',
            'message': 'communication',
            'draft': 'communication'
        }
        
        # Check for route variations
        for key, value in route_mappings.items():
            if key in route:
                route = value
                break
        
        # Validate route
        valid_routes = ['prospecting', 'insights', 'communication', 'end']
        if route not in valid_routes:
            # Default routing based on keywords
            message_lower = last_message.lower()
            if any(word in message_lower for word in ['find', 'search', 'businesses', 'companies', 'list']):
                return 'prospecting'
            elif any(word in message_lower for word in ['analysis', 'details', 'insight', 'report']):
                return 'insights'
            elif any(word in message_lower for word in ['email', 'message', 'draft', 'write']):
                return 'communication'
            else:
                return 'prospecting'  # Default to prospecting
        
        return route
        
    except Exception as e:
        print(f"Router error: {e}")
        return 'prospecting'  # Safe default