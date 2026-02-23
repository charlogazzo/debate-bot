import os
import asyncio
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event, Context
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent
)
from pydantic import BaseModel, Field

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

class DebateContext(BaseModel):
    """
    Debate context model
    """
    debate_motion: str = "AI will replace developers"
    arguments: list[str] = Field(default_factory=list)
    turn: int = 0
    max_turns: int = 6


## processing workflows are emitted from the moderating agent
class ProcessingEvent(Event):
    def __init__(self, debate_context: Context[DebateContext]):
        super().__init__()
        self.debate_context = debate_context  # contains the chat history and the most recent argument


## define the workflow between agents
class DebateWorkflow(Workflow):
    ## initialize the workflow with the agents
    def __init__(self, for_agent: ReActAgent, against_agent: ReActAgent):
        super().__init__()
        self.for_agent = for_agent
        self.against_agent = against_agent

    @step
    async def start(self, ctx: Context[DebateContext], event: StartEvent) -> ProcessingEvent:
        """
        This receives the debate motion and assigns it to the contestant agents
        :param ctx: the debate context to be used in the workflow
        :param event: StartEvent -> Triggers the start of the workflow
        :return: ProcessingEvent assigning the turn to the next contestant agent
        """

        # # initialize the context within the method that contains the start event
        # debate_context = {
        #     "motion": event.input,
        #     "arguments": [],
        #     "turn": 0,
        #     "max_turns": 6
        # }

        async with ctx.store.edit_state() as ctx_state:
            pass

        return ProcessingEvent(ctx)

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

        ctx = event.debate_context
        agent = for_agent if await ctx.store.get("turn") % 2 == 0 else against_agent

        prompt = f"""
        Motion: {await ctx.store.get("debate_motion")}
        
        Previous arguments:
        {"\n".join(await ctx.store.get("arguments"))}
        """

        response = await agent.run(prompt)

        arguments = await ctx.store.get("arguments")
        arguments.append(str(response))
        await ctx.store.set("arguments", arguments)
        await ctx.store.set("turn", await ctx.store.get("turn") + 1)

        if await ctx.store.get("turn") >= await ctx.store.get("max_turns"):
            return StopEvent(result=ctx)

        return ProcessingEvent(ctx)


# chat loop
print("Contestants ready! \n")

async def main():
    motion = input("Motion: ").strip()

    workflow = DebateWorkflow(for_agent, against_agent)
    ctx = Context(workflow)

    # create the debate context state and store it in the context instance
    ctx_state = DebateContext()
    await ctx.store.set_state(ctx_state)

    final_state = workflow.run(ctx=ctx)

    print("\n====== Debate Finished ======\n")
    print("Final state:", final_state)
    for i, arg in enumerate(await final_state.ctx.store.get("arguments"), 1):
        print(f"Argument {i}:\n{arg}\n")

if __name__ == "__main__":
    asyncio.run(main())
