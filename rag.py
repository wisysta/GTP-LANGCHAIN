# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import UnstructuredFileLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
# from langchain.vectorstores.chroma import Chroma
# from langchain.storage import LocalFileStore
# from pprint import pprint

# cache_dir = LocalFileStore("./.cache/")

# splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     separator="\n",
#     chunk_size=200,
#     chunk_overlap=30,
# )

# loader = UnstructuredFileLoader("./files/AP_01.pdf")

# docs = loader.load_and_split(text_splitter=splitter)

# embeddings = OpenAIEmbeddings()

# cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# vectorstore = Chroma.from_documents(documents=docs, embedding=cache_embeddings)
# result = vectorstore.similarity_search("유사성")

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore

cache_dir = LocalFileStore("./.cache/")


splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
loader = UnstructuredFileLoader("./files/AP_01.pdf")

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = Chroma.from_documents(docs, cached_embeddings)

result = vectorstore.similarity_search("유사성")

print(result)
