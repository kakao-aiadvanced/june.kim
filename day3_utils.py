from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# tavily client
from tavily import TavilyClient

tavily_apikey = open('tavily_api.txt', 'r').read().strip()
tavily = TavilyClient(api_key=tavily_apikey)

# openai api key
import os

openai_api_key = open('openai_api.txt', 'r').read().strip()
os.environ['OPENAI_API_KEY'] = openai_api_key

# will use llama3 model with Ollama
local_llm = 'llama3'

def get_retriever():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma-2",
        embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
    )
    retriever = vectorstore.as_retriever()
    return retriever


### Retrieval Grader
# 어떤 문서가 특정 질문과 관련이 있는지를 평가하는 모델을 생성하고 사용
# 여기서 "관련성"은 문서가 질문의 키워드를 포함하고 있는지 여부에 따라 평가됨
def get_retrieval_grader():
    # Model Initialization
    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # PromptTemplate을 사용하면 특정 양식을 갖춘 자유로운 텍스트 입력을 처리할 수 있음
    # 이 템플릿에는 두 가지 입력 변수인 'question'과 'document'가 있음
    # 이들 변수는 invoke 메소드를 통해 제공됨
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    # Retrieval Grader Pipeline Creation
    # 모델에 터미널 파이프라인을 생성함
    # 첫 번째 단계는 템플릿을 사용하여 질문과 문서를 포멧팅
    # 두 번째 단계는 이를 모델에 통과시킴
    # 마지막 단계는 결과를 파싱하여 사용가능한 형식으로 만듦
    retrieval_grader = prompt | llm | JsonOutputParser()
    return retrieval_grader


### Generate
# 사용자의 질문에 대한 답변을 생성
# 답변은 검색된 문서 내용(context)에 기반하여 생성되며,
# 만일 답변을 모를 경우 모델은 모른다고 응답하고, 답변은 최대 3문장으로 요약되어야 함
def get_generator():
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    llm = ChatOllama(model=local_llm, temperature=0)

    # Post-processing
    # 가져온 문서의 내용을 문자열로 조인하여 반환
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

### Hallucination Grader
# 생성된 답변이 주어진 사실 집합에 기반하고 있는지를 평가
# 사실에 기반한 생성을 평가하는 데에 중요
# 결과를 통해 AI의 출력이 얼마나 현실적이고 사실적인지를 판단
def get_hallucination_grader():
    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    # prompt 를 llm모델이 처리하고 JsonOutputParser를 이용하여 결과를 파싱하고 출력받음
    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader

def get_answer_grader():
    ### Answer Grader
    # 생성된 답변이 주어진 질문을 해결하는 데 유용한지를 평가하는 모델을 생성하고 사용함
    # AI의 출력이 질문에 대한 유용한 답변을 제공하고 있는지를 판단하는 데 중요함

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()

    return answer_grader

from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    상태 그래프의 상태를 정의하는 클래스

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        urls: list of url 웹검색시 관련 url 정보
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    urls: List[str]


retriever = get_retriever()
generator = get_generator()
retrieval_grader = get_retrieval_grader()
question_router = get_retrieval_grader()
answer_grader = get_answer_grader()
hallucination_grader = get_hallucination_grader()

### Nodes
# 노드 함수 정의
# 상태 그래프의 여러 노드를 정의하는 함수들
def retrieve(state):
    """
    Retrieve documents from vectorstore
    주어진 질문에 대한 문서 얻기
    얻은 문서는 상태 그래프의 documents에 추가됨

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    print(documents)
    return {
        "documents": documents,
        "question": question,
    }


def generate(state):
    """
    Generate answer using RAG on retrieved documents
    어어진 문서를 바탕으로 답변을 생성

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    urls = state["urls"]

    # RAG generation
    generation = generator.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "urls": urls,
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search
    검색된 문서들이 질문과 관련성이 있는지 판단

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }


def web_search(state):
    """
    Web search based based on the question
    웹에서 질문과 관련된 문서 얻기

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    # docs = web_search_tool.invoke({"query": question})
    docs = tavily.search(query=question)['results']
    urls = [d["url"] for d in docs]
    print("*****")
    print(urls)

    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {
        "documents": documents,
        "question": question,
        "urls": urls,
    }


### Conditional edge
def route_question(state):
    """
    Route question to web search or RAG.
    질문의 출처가 웹 검색 결과인지, 문서 검색 결과인지 결정 "websearch" or "vectorstore"

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search
    생성할 것인지, 또는 웹 검색을 포함할 것인지 결정

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
        "websearch" or "generate"
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    생성물이 문서에 기반하는지, 그리고 질문을 어떻게 처리하는지 평가

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# 상태 그래프 생성
workflow = StateGraph(GraphState)

# Define the nodes
# 앞서 정의된 노드 함수들을 그래프에 추가
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

print(workflow.nodes)

# Build graph

# 앞서 정의한 workflow 상태 그래프의 흐름을 정의
# 하나의 노드에서 다른 노드로의 연결을 설정
# 그래프의 초기 진입점 및 각 노드에서의 분기를 결정함

# Docs Retrieval 로 시작
workflow.set_entry_point("retrieve")

# Edge 추가
# add_edge 함수를 통해 두 노드 사이의 직접적인 연결을 추가함
# retrieve 노드에서 grade_documents 노드로 연결
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

# websearch 노드에서 generate 노드로 연결
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)