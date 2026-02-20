import os
import asyncio
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent
)

## retrieve token and initialize llm
hf_token = os.environ.get("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token)

## define the participating agents
for_agent = ReActAgent(tools=[],
                       model=llm,
                       add_base_tools=False,
                       system_prompt="You are a debater whose task is to argue in favour of a given motion"
                                     "Argue in a concise, professional tone."
                                     "When you are done providing your arguments say 'Over to you moderator'")

against_agent = ReActAgent(tools=[],
                           model=llm,
                           add_base_tools=False,
                           system_prompt="You are a debater whose task is to argue against a given motion"
                                         "Argue in an aggressive, dismissive tone"
                                         "When you are done providing your arguments say 'Over to you moderator'")

moderator_agent = ReActAgent(tools=[],
                             model=llm,
                             add_base_tools=False,
                             managed_agents=[for_agent, against_agent],
                             system_prompt="You are the moderator of a debate. You will receive the topic"
                                           "of the debate and pick one of the agents assigned to you to provide"
                                           "the first arguments. When the agent provides a final answer, it then becomes"
                                           "the turn of the next agent")


## processing workflows are emitted from the moderating agent
class ProcessingEvent(Workflow):
    def __init__(self, debate_context):
        super().__init__()
        self.debate_context = debate_context  # contains the chat history and the most recent argument


## define the workflow between agents
class DebateWorkflow(Workflow):
    @step
    async def receive_motion(self, event: StartEvent) -> ProcessingEvent:
        """
        This receives the debate motion and assigns it to the contestant agents
        :param event: StartEvent -> Triggers the start of the workflow
        :return: ProcessingEvent assigning the turn to the next contestant agent
        """
        debate_context = {}
        debate_context.setdefault("motion", "Talk about something interesting")
        return ProcessingEvent(debate_context)

    @step
    async def provide_argument(self, event: ProcessingEvent) -> ProcessingEvent | StopEvent:
        """
        This receives the debate motion and assigns it to the contestant agents
        The method uses the debate_motion and the most recent argument and provides this to the next
        contestant agent to formulate the next argument/response
        :param event: ProcessingEvent from the start of the workflow
        :return: ProcessingEvent assigning the turn to the next contestant agent
         or a StopEvent declaring the end of the debate
        """

        debate_motion = event.debate_context["motion"]
        most_recent_argument = event.debate_context["argument"][-1]

        ## append the most recent argument provided by the current agent
        ## The response indicated below should be the response from either of the agents
        ## whose turn it is to provide an argument
        response = None
        debate_motion["argument"].append(response)

        pass


# chat loop
print("Contestant 1 ready! \n")

user_input = input("Motion: ").strip()

async def run():
    response = await moderator_agent.run(user_input, reset=False)
    print("Bot: ", response)


if __name__ == "__main__":
    loop = asyncio.run(run())
