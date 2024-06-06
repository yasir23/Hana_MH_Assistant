import os
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time
import markdown
from langchain_community.document_loaders import UnstructuredMarkdownLoader


from typing import Any, Callable, Dict, List, Optional, Union
from langchain.docstore.document import Document
import json

from langchain.document_loaders.base import BaseLoader
from pathlib import Path

# file_path='../RAW data/json/Mhealth.json'

# file_path='combined_dataset.json'

class JSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        text_content: bool = True,
        json_lines: bool = False,
    ):
        """
        Initializes the JSONLoader with a file path, an optional content key to extract specific content,
        and an optional metadata function to extract metadata from each record.
        """
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._text_content = text_content
        self._json_lines = json_lines

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""
        docs: List[Document] = []
        if self._json_lines:
            with self.file_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._parse(line, docs)
        else:
            self._parse(self.file_path.read_text(encoding="utf-8"), docs)
        return docs
        # Print the list of documents
        for doc in docs:
            print(doc)
        
        

    def _parse(self, content: str, docs: List[Document]) -> None:
        """Convert given content to documents."""
        data = json.loads(content)

        # Perform some validation
        # This is not a perfect validation, but it should catch most cases
        # and prevent the user from getting a cryptic error later on.
        if self._content_key is not None:
            self._validate_content_key(data)
        if self._metadata_func is not None:
            self._validate_metadata_func(data)

        for i, sample in enumerate(data, len(docs) + 1):
            text = self._get_text(sample=sample)
            metadata = self._get_metadata(sample=sample, source=str(self.file_path), seq_num=i)
            docs.append(Document(page_content=text, metadata=metadata))

    def _get_text(self, sample: Any) -> str:
        """Convert sample to string format"""
        if self._content_key is not None:
            content = sample.get(self._content_key)
        else:
            content = sample

        if self._text_content and not isinstance(content, str):
            raise ValueError(
                f"Expected page_content is string, got {type(content)} instead. \
                    Set `text_content=False` if the desired input for \
                    `page_content` is not a string"
            )

        # In case the text is None, set it to an empty string
        elif isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content) if content else ""
        else:
            return str(content) if content is not None else ""

    def _get_metadata(self, sample: Dict[str, Any], **additional_fields: Any) -> Dict[str, Any]:
        """
        Return a metadata dictionary base on the existence of metadata_func
        :param sample: single data payload
        :param additional_fields: key-word arguments to be added as metadata values
        :return:
        """
        if self._metadata_func is not None:
            return self._metadata_func(sample, additional_fields)
        else:
            return additional_fields

    def _validate_content_key(self, data: Any) -> None:
        """Check if a content key is valid"""
        sample = data.first()
        if not isinstance(sample, dict):
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict), \
                    so sample must be a dict but got `{type(sample)}`"
            )

        if sample.get(self._content_key) is None:
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict) \
                    with the key `{self._content_key}`"
            )

    def _validate_metadata_func(self, data: Any) -> None:
        """Check if the metadata_func output is valid"""

        sample = data.first()
        if self._metadata_func is not None:
            sample_metadata = self._metadata_func(sample, {})
            if not isinstance(sample_metadata, dict):
                raise ValueError(
                    f"Expected the metadata_func to return a dict but got \
                        `{type(sample_metadata)}`"
                )






file_path = 'Mhealth.json'






if "vector" not in st.session_state:

    st.session_state.embeddings = FastEmbedEmbeddings(model_name = 'BAAI/bge-small-en-v1.5')

    st.session_state.loader = JSONLoader(file_path)
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents( st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
st.title("Hana (Mental Health Assistant)")


with st.chat_message("user"):
    st.write("Hello👋 How can Assist you!")



sidebar_logo = "8d33d83eac1cf8e3ea1b4840ccc8baef-removebg-preview.png"






st.logo(sidebar_logo,link=None, icon_image=None)


# "with" notation
with st.sidebar:
    st.title("Prompts")
    st.markdown("Talk to the chatbot!")
    

llm = ChatGroq(
    api_key = st.secrets["GROQ_API_KEY"],
    model_name='mixtral-8x7b-32768'
    )

prompt = ChatPromptTemplate.from_template("""
Act as the mental health bot who will council the user for mental health issues.
According to the context provided.guide him to think in present.
Use the tone of words like a girlfriend and close friend,
but not tell him I am your close friend or girlfriend just use the tone of it in your words.
use the words of love and passion.                  
                                        
I will tip you $200 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.chat_input("Input your prompt here")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start}")



    st.write(response["answer"])

    # # With a streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant chunks
    #     for i, doc in enumerate(response["context"]):
    #         # print(doc)
    #         # # st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
    #         st.write(doc.page_content)
    #         st.write("--------------------------------")
