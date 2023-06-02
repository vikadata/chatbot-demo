import logging
import json
import pinecone
import os
import flask
from flask import Flask
from tqdm.auto import tqdm
from uuid import uuid4
from datasets import load_dataset
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent


app = Flask(__name__)
app.config["DEBUG"] = True
logging.basicConfig(filename='record.log', level=logging.DEBUG)
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ['PINECONE_API_KEY'] = "YOUR_PINECONE_API_KEY"
environment = "YOUR_PINECONE_ENVIRONMENT"
index_name = "dst111-chatbot-test"
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.environ['OPENAI_API_KEY']
)

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=environment)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/datasheets/<datasheet_id>/chatbots')
def create_chatbot():
    return 'creating chatbot'

@app.route('/datasheets/<datasheet_id>/train', methods=['PUT'])
def train(datasheet_id):
    create_index(index_name)
    index = pinecone.GRPCIndex(index_name)
    app.logger.info(index.describe_index_stats())

    # data = load_dataset('squad', split='train')
    # data = load_dataset('json', data_files='./faqs.json', split='train')
    # app.logger.info('loaded dataset', "", data)
    faqs = load_faqs()
    # loop faqs
    for faq in faqs:
        metadatas = [{
            'datasheet_id': datasheet_id,
            'title': 'datasheet name',
            # 'text': 'datasheet description',
        }]
        documents = [json.dumps(faq)]
        embeds = embed.embed_documents(documents)
        ids = [faq['id']]
        # add everything to pinecone
        index.upsert(vectors=zip(ids, embeds, metadatas))

    # data = data.to_pandas()
    # # app.logger.info('converted to pandas', "", data)

    # batch_size = 100
    # for i in tqdm(range(0, len(data), batch_size)):
    #     # get end of batch
    #     i_end = min(len(data), i+batch_size)
    #     batch = data.iloc[i:i_end]
    #     # first get metadata fields for this record
    #     metadatas = [{
    #         'datasheet_id': datasheet_id,
    #         'title': 'datasheet name',
    #         'text': 'datasheet description',
    #     } for j, record in batch.iterrows()]
    #     # get the list of contexts / documents
    #     # app.logger.info('batch', '', batch)
    #     documents = batch['context']
    #     # app.logger.info('documents', '', documents)
    #     # create document embeddings
    #     embeds = embed.embed_documents(documents)
    #     # get IDs
    #     ids = batch['id']
    #     # add everything to pinecone
    #     index.upsert(vectors=zip(ids, embeds, metadatas))

    app.logger.info(index.describe_index_stats())
    return 'FAQs indexed successfully!'

# load faqs.json
def load_faqs():
    with open('faqs.json', 'r') as f:
        faqs = json.load(f)
    return faqs

# create index if not exists
def create_index(index_name):
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='dotproduct',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

@app.route('/chatbots/<chatbot_id>/similarity_search', methods=['POST'])
def similarity_search(chatbot_id):
    query = flask.request.get_json()["query"]
    app.logger.info('query', query)
    text_field = "title"
    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )
    answer = vectorstore.similarity_search(
      query,  # our search query
      k=3  # return 3 most relevant docs
    )
    app.logger.info('answer is %s', answer)
    return answer

@app.route('/chatbots/<chatbot_id>/conversation', methods=['POST'])
def conversation(chatbot_id):
    query = flask.request.get_json()["query"]
    text_field = "title"
    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    ) 
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa.run(query)

@app.route('/chatbots/<chatbot_id>/conversation_agent', methods=['POST'])
def conversation_agent(chatbot_id):
    query = flask.request.get_json()["query"]
    text_field = "title"
    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    ) 
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )
    answer = agent(query)
    return answer['output']

app.run()