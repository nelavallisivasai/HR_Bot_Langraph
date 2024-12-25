# pylint: disable = http-used,print-used,no-self-use

import datetime
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from tools import initialize_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from main import user

_ = load_dotenv()

tools, llm = initialize_tools(api_key=os.getenv("GROQ_API_KEY"))

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


TOOLS_SYSTEM_PROMPT = f"""You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. 
You have access to the tools use them whenever you need some extra information. Once you are ready with your final answer then end the process and give output. 
"""

class Agent:

    def __init__(self):
        self._tools = {t.name: t for t in tools}
        self._tools_llm = llm.bind_tools(tools)

        builder = StateGraph(AgentState)
        builder.add_node('call_tools_llm', self.call_tools_llm)
        builder.add_node('invoke_tools', self.invoke_tools)
        builder.set_entry_point('call_tools_llm')

        builder.add_conditional_edges('call_tools_llm', Agent.exists_action, {'more_tools': 'invoke_tools', 'end': END})
        builder.add_edge('invoke_tools', 'call_tools_llm')
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory)

        print(self.graph.get_graph().draw_mermaid())

    @staticmethod
    def exists_action(state: AgentState):
        print("> exists_action")
        result = state['messages'][-1]
        if len(result.tool_calls) == 0:
            return 'end'
        return 'more_tools'
    
    def call_tools_llm(self, state: AgentState):
        print("> call_tools_llm")
        messages = state['messages']
        print("input_messages: ", messages)
        messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + messages
        message = self._tools_llm.invoke(messages)
        print("Result_from_tools_llm: ", message)
        return {'messages': [message]}

    def invoke_tools(self, state: AgentState):
        print("> invoke_tools")
        tool_calls = state['messages'][-1].tool_calls
        print("tool_calls: ", tool_calls)
        results = []
        for t in tool_calls:
            print(f'Calling: {t}')
            if not t['name'] in self._tools:  # check for bad tool name from LLM
                print('\n ....bad tool name....')
                result = 'bad tool name, retry'  # instruct LLM to retry if bad
            else:
                result = self._tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("tool_results: ", results)
        print('Back to the main model!')
        return {'messages': results}