# agents/insights.py  
from .base import create_agent, agent_node
from graph.state import GraphState

def create_insights_agent(llm, insights_tools):
    """Create insights agent meeting BuzzBoard criteria requirements."""
    return create_agent(
        llm, 
        insights_tools, 
        "You are a BuzzBoard insights analyst. Use get_prospect_details for deep prospect analysis.\n\n"
        "FORMAT (Max 200 words):\n\n"
        "**[Business Name] - [Category]**\n"
        "Location: [City, State] | Digital Score: [X/6]\n\n"
        "**METRICS ANALYSIS**\n"
        "SEO Score: [Poor/Good based on digital presence]\n"
        "Social Presence: [Active channels or 'Weak']\n"
        "Local SEO: [Missing/Present - check Google presence]\n\n"
        "**SWOT ANALYSIS**\n"
        "S: [1 key strength from active channels]\n"
        "W: [1 critical weakness from missing channels] \n"
        "O: [1 market opportunity from gaps]\n"
        "T: [1 competitive threat]\n\n"
        "**VALUE PROPOSITION**\n"
        "[Specific benefit for their industry + location]\n\n"
        "**ENGAGEMENT STRATEGY**\n"
        "Approach: [How to contact - email/phone/in-person]\n"
        "Hook: [Opening conversation starter]\n"
        "Focus: [Primary service to discuss]\n\n"
        "INSTRUCTIONS:\n"
        "- Flag missing local SEO if no Google presence\n"
        "- Tailor value prop to industry needs\n"
        "- Recommend engagement based on business type\n"
        "- Keep analysis concise but actionable"
    )

def insights_node(state: GraphState, agent): 
    return agent_node(state, agent, "InsightsAgent")