# Schemas for formatting answers
from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """ Answer the questions. """

    answer: str = Field(
        description="10 word answer to the question.")
    reflection: Reflection = Field(
        description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1 search query for researching improvements to address the critique of your current answer.")


class ReviseAnswer(AnswerQuestion):
    """ Revise your original answer to your question. """

    references: List[str] = Field(
        description="Citations motivating your updated answer.")
