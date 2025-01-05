import streamlit as st
import os
import textract
from langchain.chat_models import ChatOpenAI
from itertools import zip_longest
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import HuggingFaceHub
import emoji
from langchain.utilities import SerpAPIWrapper



# Set the OpenAI API key
OPENAI_API_KEY =""

# Define your directory containing PDF files here
pdf_dir = 'career_bot'
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)



def get_response(history,user_message,temperature=0):
    print('called')
    DEFAULT_TEMPLATE = """As an AI-powered digital journalist, you have an expertise in comprehending, summarizing, and delivering information sourced 
        from reputable news outlets. You maintain a firm grasp on current trends and hot news topics, providing users with verified and unbiased insights 
        in a conversational style. The user will interact with you to learn about the latest headlines, getting informed about trending topics and stories 
        they are interested in. In every interaction, your focus is to provide information that is accurate, timely, and clear.
    It follows the previous conversation to do so

    Relevant pieces of previous conversation:
    {context},


    Useful news information from Web:
    {web_knowledge},

    Current conversation:
    Human: {input}
    News Journalist:"""

    PROMPT = PromptTemplate(
        input_variables=['context','input','web_knowledge'], template=DEFAULT_TEMPLATE
    )



    params = {
    "engine": "bing",
    "gl": "us",
    "hl": "en",
    }

    search = SerpAPIWrapper(params=params)

    web_knowledge=search.run(user_message)



  
    chat_gpt = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo-16k",openai_api_key=OPENAI_API_KEY)

    conversation_with_summary = LLMChain(
        llm=chat_gpt,
        prompt=PROMPT,
        verbose=True
    )
    response = conversation_with_summary.predict(context=history,input=user_message,web_knowledge=web_knowledge)
    return response

# Function to get conversation history
def get_history(history_list):
    history = ''
    for message in history_list:
        if message['role']=='user':
            history = history+'input '+message['content']+'\n'
        elif message['role']=='assistant':
            history = history+'output '+message['content']+'\n'
    
    return history


# Streamlit UI
def get_text():
    input_text = st.sidebar.text_input("You: ", "Hello, how are you?", key="input")
    if st.sidebar.button('Send'):
        return input_text
    return None

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

user_input = get_text()

if user_input:
    user_history = list(st.session_state["past"])
    bot_history = list(st.session_state["generated"])

    combined_history = []
    for user_msg, bot_msg in zip_longest(user_history, bot_history):
        if user_msg is not None:
            combined_history.append({'role': 'user', 'content': user_msg})
        if bot_msg is not None:
            combined_history.append({'role': 'assistant', 'content': bot_msg})

    formatted_history = get_history(combined_history)

    output = get_response(formatted_history,user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Chat History", expanded=True):
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            st.markdown(emoji.emojize(f":speech_balloon: **User {str(i)}**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant {str(i)}**: {st.session_state['generated'][i]}"))