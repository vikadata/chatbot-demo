# Chatbot with Flask and Pinecone

This is a simple chatbot application that uses Flask and Pinecone. It allows users to ask questions and get answers from a knowledge base.

Referred to Pinecone's example [Chatbot Agents with LangChain](https://docs.pinecone.io/docs/langchain-retrieval-agent), more examples: [Examples](https://docs.pinecone.io/docs/examples?utm_medium=email&_hsmi=250250907&_hsenc=p2ANqtz-9i0hjQUTYllnr6M_YMZz4-XAubyWt02yAwFNT640JnWagcxPnrXWzJXsWJv4rZW_vi56mRFbB0A4t_yZ9KhzG_rFs8Ag&utm_content=248662537&utm_source=hs_automation)

## Getting Started

### Create Pinecone Accout
Sign up free in [Pinecone](https://www.pinecone.io/)

### Installation
1. create and activate the environment
```
python3 -m venv .venv
. .venv/bin/activate
```
2. Install the dependencies:
```
pip install Flask
!pip install -qU openai pinecone-client[grpc] langchain==0.0.162 tiktoken datasets
```
if you got error while installing pinecone-client[grpc], use the following command instead:
```
pip3 install -U "pinecone-client[grpc]" install -qU openai "pinecone-client[grpc]" langchain==0.0.162 tiktoken datasets
```

### Quick start
1. Use your own API key, as changing machines or networks can easily result in getting banned
replace the following API key with your own API key
```
YOUR_OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
YOUR_PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
YOUR_PINECONE_ENVIRONMENT = "YOUR_PINECONE_ENVIRONMENT"
```
2. Start the Flask application
```
flask --app chatbot run
```

The chatbot will now be running on http://localhost:5000/.

3. Training
```
curl --location --request PUT 'http://127.0.0.1:5000/datasheets/dst111/train' \
```

4. Querying
```
curl --location --request POST 'http://127.0.0.1:5000/chatbots/cb111/conversation_agent' \
--header 'Content-Type: application/json' \
--data-raw '{
    "query": "What is APITable?"
}'
```
you could change the query parameter if you have another question



