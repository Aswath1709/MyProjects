from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mykey import openai_key
import os
from langchain_core.output_parsers import StrOutputParser


os.environ['OPENAI_API_KEY'] = openai_key

# Initialize the LLM
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7)

# Initialize the output parser
output_parser = StrOutputParser()

def response_from_llm(user_question, chat_history, context):
    # Refined prompt template to ensure accurate responses
    template_chat = '''
  You are an AI-powered business intelligence assistant.

## Chat History:
{chat_history}

## Context:
{context}

## User Question:
{user_question}

## Guidelines:
- Answer based on both the **chat history and the provided context**.
- If the user's question is related to **past conversations**, use chat history to recall details (e.g., their name, previous questions, preferences).
- If the question is **directly related to the provided context**, answer using the available information.
- If information is present, no need to specify where it came from (eg. Based on document provided, Based on chat history provided)
- If neither chat history nor context contains relevant details, say:
  - *"The provided documents and chat history do not contain enough information to answer that."*
- Maintain **concise yet insightful responses** while ensuring factual accuracy.
    '''

    # Create the ChatPromptTemplate with the defined prompt
    prompt = ChatPromptTemplate.from_template(template_chat)

    # Use LLMChain for chaining the prompt and LLM
    chain = prompt | llm | output_parser

    # Use the .stream() method for streaming responses
    response = chain.stream({'chat_history': chat_history, 'user_question': user_question, 'context': context})

    return response
