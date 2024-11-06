from langchain_elasticsearch import ElasticsearchStore, SparseVectorStrategy, ElasticsearchRetriever
from elasticsearch.helpers.vectorstore import SparseVectorStrategy
from typing import Optional, List, Dict, Any
from llm_integrations import get_llm
from elasticsearch_client import (
    elasticsearch_client,
    get_elasticsearch_chat_message_history,
)
from flask import render_template, stream_with_context, current_app
import json
import os

# INDEX = os.getenv("ES_INDEX", "workplace-app-docs")
INDEX_CHAT_HISTORY = os.getenv(
    "ES_INDEX_CHAT_HISTORY", "frontend-chat-history"
)
ELSER_MODEL = os.getenv("ELSER_MODEL", ".elser_model_2")
SESSION_ID_TAG = "[SESSION_ID]"
SOURCE_TAG = "[SOURCE]"
DONE_TAG = "[DONE]"

# store = ElasticsearchStore(
#     es_connection=elasticsearch_client,
#     index_name=INDEX,
#     strategy=SparseVectorStrategy(model_id=ELSER_MODEL),
# )


def build_query(query):
    return {
        "retriever": {
            "standard": {
                "query": {
                    "sparse_vector": {
                        "field": "ml.inference.body_expanded.predicted_value",
                        "inference_id": ".elser_model_2_linux-x86_64",
                        "query": query
                    }
                }
            }
        },
        "size": 3
    }

@stream_with_context
def ask_question(question, session_id, index):
    yield f"data: {SESSION_ID_TAG} {session_id}\n\n"
    current_app.logger.debug("Chat session ID: %s", session_id)

    chat_history = get_elasticsearch_chat_message_history(
        INDEX_CHAT_HISTORY, session_id
    )

    if len(chat_history.messages) > 0:
        # create a condensed question
        condense_question_prompt = render_template(
            "condense_question_prompt.txt",
            question=question,
            chat_history=chat_history.messages,
        )
        condensed_question = get_llm().invoke(condense_question_prompt).content
    else:
        condensed_question = question

    current_app.logger.debug("Condensed question: %s", condensed_question)
    current_app.logger.debug("Question: %s", question)

    # Move this code to the function so I can pass index dynamically
    index_source_fields = {
        index: "body"
    }

    retriever = ElasticsearchRetriever(
        index_name=index,
        body_func=build_query,
        content_field=index_source_fields,
        es_client=elasticsearch_client
    )

    # docs = store.as_retriever().invoke(condensed_question)
    docs = retriever.invoke(condensed_question)
    for doc in docs:
        doc_source = {**doc.metadata, "page_content": doc.page_content}
        current_app.logger.debug(
            "Retrieved document passage from: %s", doc.metadata["_source"]["name"]
        )
        yield f"data: {SOURCE_TAG} {json.dumps(doc_source)}\n\n"

    qa_prompt = render_template(
        "rag_prompt.txt",
        question=question,
        docs=docs,
        chat_history=chat_history.messages,
    )

    answer = ""
    for chunk in get_llm().stream(qa_prompt):
        content = chunk.content.replace(
            "\n", " "
        )  # the stream can get messed up with newlines
        yield f"data: {content}\n\n"
        answer += chunk.content

    yield f"data: {DONE_TAG}\n\n"
    current_app.logger.debug("Answer: %s", answer)

    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)
