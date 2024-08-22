from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class AskBot:
    """The QnA bot based on OpenAI and Chroma, with Langchain for tooling."""

    def __init__(self, file_path: str, system_message: str) -> None:
        self.file_path = file_path
        self.system_message = system_message
        self.loader = TextLoader(self.file_path, encoding="UTF-8")
        self.documents = self.loader.load()
        self.texts = self.split_text(self.documents)
        self.vectordb = Chroma.from_documents(
            self.texts,
            OpenAIEmbeddings(),
        )

        prompt_template = f"""
        {system_message}

        Human: {{question}}

        AI: Let's approach this step-by-step:

        1) First, I'll review the relevant parts of the codebase.
        2) Then, I'll provide a clear and concise answer based on the repository's content.
        3) If necessary, I'll explain any complex concepts in simple terms.
        4) Finally, I'll suggest where in the codebase you might look for more information, if applicable.

        Here's the answer to your question:

        {{context}}

        Based on this information from the repository:
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.bot = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
        )

    def split_text(self, documents: TextLoader):
        """Generic splitter for separating text into chunks for related retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    def query(self, query: str):
        """The actual QnA step following the invoke command."""
        return self.bot.invoke(query)
