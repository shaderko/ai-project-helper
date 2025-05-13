import os
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
import argparse
import sys

MODEL_NAME = "qwen3-32:4b"
DOC_DIR = ""

EXCLUDED_DIRS = {".git", "node_modules"}


def is_excluded(path):
    for excluded in EXCLUDED_DIRS:
        if f"/{excluded}/" in path or path.endswith(f"/{excluded}"):
            return True
    return False


def load_filtered_documents(base_path: str) -> list[Document]:
    documents = []
    for root, dirs, files in os.walk(base_path):
        # Filter excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            full_path = os.path.join(root, file)
            print(f"trying to load {full_path}")
            if not is_excluded(full_path):
                try:
                    loader = TextLoader(full_path, autodetect_encoding=True)
                    documents.extend(loader.load())
                    print(f"Loaded {full_path}")
                except Exception as e:
                    print(f"Failed to load {full_path}: {e}")
    return documents


def ingest_docs(doc_dir: str, persist_dir: str) -> Chroma:
    print(f"documents path {doc_dir}, presiting to {persist_dir}")
    docs = load_filtered_documents(doc_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    vectordb.add_documents(chunks)
    vectordb.persist()
    return vectordb


def get_directory_structure(start_path):
    structure = []
    for root, dirs, files in os.walk(start_path):
        # Skip unwanted directories
        dirs[:] = [d for d in dirs if d not in {"node_modules", ".git"}]

        rel_root = os.path.relpath(root, start_path)
        rel_root_display = "." if rel_root == "." else rel_root + "/"
        structure.append(rel_root_display)

        for f in files:
            file_path = os.path.join(rel_root, f) if rel_root != "." else f
            structure.append(f"    {file_path}")
    return "\n".join(structure)


def create_agent_with_retrieval(vectordb: Chroma, DOC_DIR: str):
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    dir_structure = get_directory_structure(DOC_DIR)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print(dir_structure)

    memory.save_context({"input": "directory structure"}, {"output": dir_structure})

    def retrieve_fn(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        memory.save_context(
            {"input": query},
            {"output": "\n---\n".join([doc.page_content for doc in docs])},
        )
        return "\n---\n".join([doc.page_content for doc in docs])

    project_info_tool = Tool(
        name="ProjectInfoRetriever",
        func=retrieve_fn,
        description=(
            "Use this tool to retrieve high-level information about the project, such as authentication methods, routing structure, "
            "technologies used, or architectural patterns. It uses semantic search to find relevant context."
        ),
    )

    def file_reader_fn(rel_path: str) -> str:
        abs_path = os.path.join(DOC_DIR, rel_path.strip())
        if not abs_path.startswith(os.path.abspath(DOC_DIR)):
            return "Error: Access to this path is not allowed."
        if not os.path.exists(abs_path):
            return f"Error: File '{rel_path}' not found."
        if os.path.isdir(abs_path):
            return f"Error: '{rel_path}' is a directory, not a file."
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            memory.save_context({"input": f"read {rel_path}"}, {"output": content})
            return content
        except Exception as e:
            return f"Error reading file '{rel_path}': {str(e)}"

    file_reader_tool = Tool(
        name="FileReader",
        func=file_reader_fn,
        description=(
            "Use this tool to read the full content of a specific file. Provide a relative file path from the project root "
            "(e.g., 'src/index.ts' or 'pages/api/auth.ts'). This is helpful when inspecting exact code in specific files."
        ),
    )

    template = (
        "You are an expert coding assistant create production ready functional code. Always respond with code changes or implementations, only if it is required respond with text.\n\n"
        "You have access to the following tools:\n\n"
        "🧾 **ProjectInfoRetriever** – Retrieves general or relevant information from the project codebase using semantic search. Use this to understand architecture, naming conventions, or find general references to topics like 'authentication', 'middleware', etc.\n"
        "📄 **FileReader** – Reads the full content of a specific file path relative to the project root (as shown in the directory structure).\n\n"
        "‼️ Important:\n"
        "- Use the FileReader tool to create a response based on the content of a specific file.\n"
        "- Use **ProjectInfoRetriever** when the user asks about how something is implemented or wants conceptual understanding (e.g., routing, error handling, framework usage).\n"
        "- Use **FileReader** if the user mentions a specific file (e.g., 'open src/auth/index.ts') or you need to inspect code line-by-line.\n"
        "- Before answering any question, always **think step-by-step about what information you need**.\n"
        "- Fetch information first — avoid reasoning without real context.\n"
        "- When in doubt, **use the tools before answering**.\n\n"
        "📌 Example (project info):\n"
        "User: How is authentication handled?\n"
        "Thought: I should retrieve general information about authentication in the project.\n"
        "Action: ProjectInfoRetriever('authentication, login, auth middleware')\n"
        "Observation: {observation}\n"
        "...\n"
        "Final Answer: [your explanation based on that info]\n\n"
        "📌 Example (file read):\n"
        "User: What’s in `src/pages/api/auth.ts`?\n"
        "Thought: I should read the content of that file.\n"
        "Action: FileReader('src/pages/api/auth.ts')\n"
        "Observation: {observation}\n"
        "...\n"
        "Final Answer: [explanation based on that file’s contents]\n\n"
        "🔁 You can repeat retrieval steps if needed.\n"
        "🧠 Think before you answer. Fetch context before you reason.\n\n"
        "User Input: {input}\n"
        "{agent_scratchpad}"
    )

    prompt = PromptTemplate(
        template=template, input_variables=["input", "observation", "agent_scratchpad"]
    )

    llm = Ollama(
        model=MODEL_NAME,
        temperature=0.7,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    agent = initialize_agent(
        tools=[project_info_tool, file_reader_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        prompt=prompt,
    )
    return agent


def main():
    parser = argparse.ArgumentParser(description="RAG Coding Assistant")
    parser.add_argument(
        "--doc-dir",
        default="/home/xd/Documents/_pulsar/customer_projects/codeforge/forger-marketplace",
        help="Directory containing documents to ingest (default: hardcoded path)",
    )
    parser.add_argument(
        "--persist-dir",
        default="vector_db",
        help="Directory to persist the vector database (default: 'vector_db')",
    )
    args = parser.parse_args()

    DOC_DIR = args.doc_dir
    print(args.doc_dir)
    PERSIST_DIR = args.persist_dir

    if not os.path.exists(DOC_DIR):
        print(f"Error: The document directory '{DOC_DIR}' does not exist.")
        sys.exit(1)

    # if not os.path.exists(PERSIST_DIR):
    #     print(
    #         f"Note: The persistence directory '{PERSIST_DIR}' does not exist. It will be created."
    #     )
    #     os.makedirs(PERSIST_DIR, exist_ok=True)

    if not os.path.exists(PERSIST_DIR):
        vectordb = ingest_docs(DOC_DIR, PERSIST_DIR)
    else:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    agent = create_agent_with_retrieval(vectordb, DOC_DIR)

    print("Welcome to the RAG Coding Assistant. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = agent.invoke(input=query)
        print("Assistant:", result)
    print("Goodbye!")


if __name__ == "__main__":
    main()
