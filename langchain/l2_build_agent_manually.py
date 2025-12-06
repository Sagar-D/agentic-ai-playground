from tools.math_tools import MATH_TOOLS
from langchain.tools import BaseTool
from langchain.chat_models import BaseChatModel
from helpers.llm_manager import get_llm_instance
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    HumanMessage,
    BaseMessage,
)


class Agent:

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        self.llm = llm.bind_tools(tools=tools)
        self.tools = tools

    def invoke(self, chat_history: list[BaseMessage]) -> AIMessage:

        print("--"*30)
        print(f"LLM INVOKED WITH {len(chat_history)} messages!!")
        print("--"*30)
        
        response = self.llm.invoke(chat_history)
        return response


class AgentExecutor:

    def __init__(self, agent: Agent, tools: list[BaseTool]):
        self.agent = agent
        self.tools = tools
        self.toolmap = {tool.name: tool for tool in tools}
        self.chat_history = []

    def invoke(self, prompt: str) -> str:

        print("--"*30)
        print(f"AGENT EXECUTOR INVOKED WITH PROMPT - {prompt}.")
        print("--"*30)

        if len(prompt.strip()) > 0 :
            self.chat_history.append(HumanMessage(prompt))

        agent_response = self.agent.invoke(self.chat_history)
        self.chat_history.append(agent_response)

        if len(agent_response.tool_calls) > 0 :
            tool_responses = self._invoke_tools(agent_response.tool_calls)
            for tool_resp in tool_responses :
                self.chat_history.append(
                    ToolMessage(content=tool_resp["content"], tool_call_id=tool_resp["tool_call_id"])
                )
            return self.invoke("")
        else :
            print("--"*30)
            print("Agent loop completed!!!")
            print("--"*30)
            return agent_response.content


    def _invoke_tools(self, tool_requests: list[dict]):

        tool_responses = []
        for tool_req in tool_requests:
            print("--"*30)
            print(f"TOOL CALL REQUESTED - { { tool_req['name'] : tool_req['args'] }}")
            print("--"*30)
            tool_resp = self.toolmap[tool_req["name"]].invoke(tool_req["args"])
            tool_responses.append({"content": tool_resp, "tool_call_id": tool_req["id"]})
            print("--"*30)
            print(f"TOOL RESPONSE - {tool_resp}")
            print("--"*30)

        return tool_responses


if __name__ == "__main__":

    llm = get_llm_instance()
    agent = Agent(llm=llm, tools=MATH_TOOLS)
    executor = AgentExecutor(agent=agent, tools=MATH_TOOLS)

    while True :
        prompt = input("\n\nUser Query : ")
        response = executor.invoke(prompt=prompt)
        print(f"\n\nAgent Response : {response}\n\n")
        
