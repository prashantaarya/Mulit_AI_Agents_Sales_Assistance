# graph/state.py
from typing import Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    messages: List[BaseMessage]
    prospects: Optional[List[Dict]] = None
    prospect_details: Optional[Dict] = None
    communication_draft: Optional[str] = None
    conversation_history: Optional[List[Dict]] = None
    user_context: Optional[Dict] = None