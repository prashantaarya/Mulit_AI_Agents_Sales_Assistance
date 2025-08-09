# graph/workflow.py
from langgraph.graph import StateGraph, END
from graph.state import GraphState

def create_workflow(prospecting_node_func, insights_node_func, communication_node_func, route_requests_func):
    """Create and compile the workflow graph."""
    workflow = StateGraph(GraphState)

    # Add the worker nodes
    workflow.add_node("prospecting", prospecting_node_func)
    workflow.add_node("insights", insights_node_func)
    workflow.add_node("communication", communication_node_func)

    # The entry point is a conditional edge that calls the router.
    workflow.set_conditional_entry_point(
        route_requests_func,
        {
            "prospecting": "prospecting",
            "insights": "insights",
            "communication": "communication",
            "end": END,
        },
    )

    # After a specialist agent has done its work, the graph's turn is over.
    workflow.add_edge("prospecting", END)
    workflow.add_edge("insights", END)
    workflow.add_edge("communication", END)

    # Compile the graph into a runnable app
    return workflow.compile()