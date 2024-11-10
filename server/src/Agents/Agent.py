from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from Tools import Tools
from Chains import Chain

class Agent:
    def __init__(self):
        self.chain = Chain()
        self.tools = Tools()

    def invoke(self, query):
        llm = self.chain.get_llm()
        tools = self.tools.get_all_tools()
        
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = agent_executor.invoke({"input": query})
        return response