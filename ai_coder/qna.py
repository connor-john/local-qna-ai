from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class AskBot:
    """"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.loader = TextLoader(self.file_path)
        self.documents = self.loader.load()
        self.texts = self.split_text(self.documents)
        self.vectordb = Chroma.from_documents(
            self.texts,
            OpenAIEmbeddings(),
        )
        self.bot = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
        )

    def split_text(self, documents: TextLoader):
        """"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    def query(self, query: str):
        """"""
        return self.bot.invoke(query)
