import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile")

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
            4. Keep your responses short and concise under 100 words.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", """ Answer the user's question above using the required format. """)
    ]
).partial(time=lambda: datetime.now().isoformat())


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed 100 words answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion")

revise_instruction = """ Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does count towards the word limit of 10 words). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 10 words. """

revisor = actor_prompt_template.partial(first_instruction=revise_instruction) | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer")

# if __name__ == "__main__":
#     human_message = HumanMessage(
#         content=""" Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital. """
#     )

#     chain = (
#         first_responder_prompt_template | llm.bind_tools(
#             tools=[AnswerQuestion], tool_choice="AnswerQuestion") | parser_pydantic
#     )

#     res = chain.invoke(input={"messages": [human_message]})
#     print(dict(res[0]))
