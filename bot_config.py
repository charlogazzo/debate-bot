from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# initialize the llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

##### ---------- PROMPTS ----------- #####


## define the pro agent system prompt
pro_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert debater arguing for the following motion:
    
    "{topic}"
    
    Your job:
    - Strongly support the motion
    - Provide clear reasoning and examples
    - Respond directly to your opponent's arguments
    
    Opponent's last argument:
    {opponent_argument}
    
    Your response:
    """
)


# define the con agent system prompt
con_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert debater arguing against the following motion:
    
    "{topic}"
    
    Your job:
    - Critically attack the motion
    - Point out weaknesses in the opponent's argument
    - Provide counterexamples
    - Use ad-hominem attacks against your opponent
    
    Opponent's last argument:
    {opponent_argument}
    
    Your response:
    """
)


# define the debate judge system prompt
judge_prompt = ChatPromptTemplate.from_template(
    """
    You are a judge evaluating a debate.
    
    Topic:
    {topic}
    
    Pro_arguments:
    {pro_history}
    
    Con arguments:
    {con_history}
    
    Decide:
    1. Who won (Pro or Con)
    2. Why (focus on logic, evidence, and rebuttals)
    """
)

####### Agent functions ########

## configuring the agents by giving them the topic and arguments in order to formulate a response

def pro_agent(topic, opponent_argument):
    chain = pro_prompt | llm
    return chain.invoke({
        "topic": topic,
        "opponent_argument": opponent_argument
    }).content

def con_agent(topic, opponent_argument):
    chain = con_prompt | llm
    return chain.invoke({
        "topic": topic,
        "opponent_argument": opponent_argument
    }).content

def judge_agent(topic, pro_history, con_history):
    chain = judge_prompt | llm
    return chain.invoke({
        "topic": topic,
        "pro_history": pro_history,
        "con_history": con_history
    }).content

######## debate loop ########

def run_debate(topic, rounds=3):
    pro_history = []
    con_history = []

    pro_last = "No argument yet."
    con_last = "No argument yet."

    for i in range(rounds):
        print(f"\n--- Round {i+1} ---")

        pro_response = pro_agent(topic, con_last)
        pro_history.append(pro_response)
        print("\nPRO:\n", pro_response)

        con_response = con_agent(topic, pro_last)
        con_history.append(con_response)
        print("\nCON:\n", con_response)

        pro_last = pro_response
        con_last = con_response

    print("\n Judgement")
    result = judge_agent(topic, "\n".join(pro_history), "\n".join(con_history))
    print(result)

###### RUN ######

if __name__ == "__main__":
    topic = "AI will do more good than harm to society"
    run_debate(topic, rounds=3)

