import os
from tqdm import tqdm
from rich.style import Style
from collections import deque
from functools import partial
from dotenv import load_dotenv
from rich.console import Console
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.runnables import RunnableLambda, RunnableAssign, RunnablePassthrough


load_dotenv()
print(os.getenv('NVIDIA_API_KEY'))
os.environ["NVIDIA_API_KEY"] = os.getenv('NVIDIA_API_KEY')


console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)


def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))


class WikipediaDocumentProcessor:
    def __init__(self, query, load_max_docs=10, chunk_size=800, chunk_overlap=100):
        self.query = query
        self.load_max_docs = load_max_docs
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = []
        self.all_chunks = []
        self.embeddings_model = CohereEmbeddings(
            cohere_api_key=os.getenv('COHERE_API_KEY'))
        self.vectorstore = None

    def load_documents(self):
        self.docs = list(tqdm(WikipediaLoader(
            query=self.query, load_max_docs=self.load_max_docs).load(), desc="Loading docs"))
        print(f"{len(self.docs)} documents loaded from Wikipedia.")

    def create_chunks_with_headers(self, doc, doc_index):
        chunks = []
        start = 0
        doc_content = doc.page_content
        doc_length = len(doc.page_content)

        while start < doc_length:
            end = min(start + self.chunk_size, doc_length)
            chunk = doc_content[start:end]

            if start != 0:
                chunk = doc_content[max(start - self.chunk_overlap, 0):end]

            chunk_json = {
                "meta_data": {
                    "title": doc.metadata["title"],
                    "summary": doc.metadata['summary'],
                    "source_url": doc.metadata['source'],
                },
                "chunk_index": len(chunks) + 1,
                "doc_index": doc_index,
                "content": chunk
            }
            chunks.append(chunk_json)

            start += self.chunk_size

        return chunks

    def process_and_create_embeddings(self):
        self.all_chunks = []
        for i, doc in enumerate(self.docs):
            chunks = self.create_chunks_with_headers(doc, i + 1)
            self.all_chunks.extend(chunks)

        documents = [Document(page_content=chunk["content"],
                              metadata=chunk["meta_data"]) for chunk in self.all_chunks]

        self.vectorstore = Chroma.from_documents(
            documents, self.embeddings_model)
        print("All data has been processed and stored in Chroma.")

    def run(self):
        self.load_documents()
        self.process_and_create_embeddings()
        print("All data has been processed.")


class CricketAssistant:
    def __init__(self, vectorstore):
        self.embeddings_model = CohereEmbeddings(
            cohere_api_key=os.getenv('COHERE_API_KEY'))
        self.vectorstore = vectorstore
        self.initialize_retriever()
        self.memory = deque(maxlen=5)
        self.llm = ChatNVIDIA(
            model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()

    def initialize_retriever(self):
        metadata_field_info = [
            AttributeInfo(
                name="title", description="The name of the article", type="string"),
            AttributeInfo(
                name="summary", description="The short summary of the article contents", type="integer"),
            AttributeInfo(
                name="source_url", description="The web uri link to the article webpage", type="string"),
        ]
        document_content_description = "Data about cricket"
        llm = ChatNVIDIA(
            model="mistralai/mistral-7b-instruct-v0.2") | StrOutputParser()
        self.retriever = SelfQueryRetriever.from_llm(
            llm, self.vectorstore, document_content_description, metadata_field_info)

    def update_memory(self, user_question, response):
        self.memory.append({"question": user_question, "response": response})

    def generate_embeddings(self, input_data):
        pprint(input_data)
        embeddings = self.retriever.invoke(input_data)
        if embeddings:
            return embeddings[:2]
        return "No data available"

    def generate_embeddings_query(self, input_data):
        prompt = ChatPromptTemplate.from_template("""
User's Question: {{input}}
Previous conversation memory {{memory}}
Generate only a query sentence and nothing else from the user's question to fetch from the data from embeddings. If the user's question does not have enough context then create a query based on the Knowledge Base.
""")
        embedding_chain = prompt | self.llm
        embeddings_query = embedding_chain.invoke(input_data)
        if embeddings_query:
            return embeddings_query
        else:
            return "Process failed"

    def get_response(self, prompt):
        response = self.llm.invoke(prompt)
        return response

    def handle_user_input(self, user_input):
        sys_msg = """
You are an intelligent assistant that answers all questions about cricket using contextual information from Wikipedia. Your responses should be conversational and informative, providing clear and concise explanations. When relevant, include the source URL of the articles to give users additional reading material.

Always aim to:
1. Answer the question directly and clearly.
2. Provide context and background information when useful but do not give irrelevant information and answer to the point.
3. Suggest related topics or additional points of interest.
4. Be polite and engaging in your responses.
5. Remove the unnecessary context from the context provided if irrelevant to the question

Now, let's get started!
"""

        Runnable = (
            {"input": RunnablePassthrough(), "memory": RunnablePassthrough()}
            | RunnableAssign({"embedding_query": RunnableLambda(self.generate_embeddings_query)})
            | RunnableAssign({"context": RunnableLambda(self.generate_embeddings)})
            | RunnableAssign({"prompt": lambda x: ChatPromptTemplate.from_template(
                f"""
{sys_msg}

User's Question: {{input}}

Context Information: {{context}}

Previous Conversation memory: {{memory}}

Your Response:
"""
            )})
            | PPrint()
            | RunnableAssign({"response": lambda x: self.get_response(x["prompt"])})
            | RunnableAssign({"memory": lambda x: self.update_memory(x["input"]["input"], x["response"])})
        )

        response = Runnable.invoke(
            {"input": user_input, "memory": self.memory})
        return response["response"]


# processor = WikipediaDocumentProcessor(
#     query="Cricket and everything related to cricket")
# processor.run()

# assistant = CricketAssistant(processor.vectorstore)

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         print("Exiting chat...")
#         break
#     assistant.handle_user_input(user_input)
