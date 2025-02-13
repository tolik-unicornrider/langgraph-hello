from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_aws import ChatBedrockConverse, ChatBedrock
from langgraph.checkpoint.memory import InMemorySaver


# Load environment variables
load_dotenv()

nova_model = "amazon.nova-micro-v1:0"
claude = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Initialize model and memory
model = ChatBedrock(
    model=nova_model,
    temperature=0
)
memory = InMemorySaver()

# Define the state structure
class AgentState(TypedDict):
    question: str
    answer: str
    reflection: str
    revision_number: int
    max_revisions: int

# Define prompts
ANSWER_PROMPT = """You are a helpful AI assistant. Provide a clear and informative answer 
to the user's question."""

REFLECTION_PROMPT = """You are a critical thinker. Review the following answer and suggest 
improvements for clarity, accuracy, and completeness. If the answer is already excellent, 
say 'No improvements needed.'"""

# Define nodes
def answer_node(state: AgentState):
    """Generate an initial answer or revised answer."""
    messages = [
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content=state['question'])
    ]
    response = model.invoke(messages)
    return {
        **state,  # Preserve existing keys, including "configurable"
        "answer": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    """Reflect on the current answer."""
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['answer'])
    ]
    response = model.invoke(messages)
    return {
        **state,  # Preserve the configurable keys and other state data
        "reflection": response.content
    }

def should_continue(state):
    """Determine if we should continue refining the answer."""
    if state["revision_number"] > state["max_revisions"]:
        return END
    if "No improvements needed" in state.get("reflection", ""):
        return END
    return "reflect"

# Build the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("generate_answer", answer_node)
builder.add_node("reflect", reflection_node)

# Set entry point
builder.set_entry_point("generate_answer")

# Add edges
builder.add_conditional_edges(
    "generate_answer",
    should_continue,
    {END: END, "reflect": "reflect"}
)
builder.add_edge("reflect", "generate_answer")

# Compile the graph
graph = builder.compile() #checkpointer=memory)

if __name__ == "__main__":
    # Example usage with the required key added
    config = {
        "question": "What is the capital of France?",
        "max_revisions": 2,
        "revision_number": 1,
        "configurable": {"thread_id": 1}
    }
    
    # Run the graph and print each state
    for state in graph.stream(config):
        print("\nState:", state)
