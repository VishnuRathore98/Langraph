import os
from pprint import pprint
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from schemas import AnswerQuestion
from langchain_openai import ChatOpenAI

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
# openai_key = os.getenv("OPEN_AI_KEY")

llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile")
# llm = ChatOpenAI(api_key=openai_key, model="gpt-4o")

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# Chains for model
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher.
            Current time: {time}
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", """ Answer the user's question above using the required format. """)
    ]
).partial(time=lambda: datetime.now().isoformat())


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~200 words answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion")

if __name__ == "__main__":
    human_message = HumanMessage(
        content=""" Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital. """
    )

    chain = (
        first_responder_prompt_template | llm.bind_tools(
            tools=[AnswerQuestion], tool_choice="AnswerQuestion") | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(dict(res[0]))
