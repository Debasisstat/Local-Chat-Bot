
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def create_chain(vectorstores, model_name, temp, prompt):
    """Creates a conversational chain with a system prompt."""
    llm = Ollama(model=model_name, temperature=temp)
    retriever = vectorstores.as_retriever()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    # Inject system prompt into memory
    memory.save_context({"input": "System Prompt"}, {"output": prompt})

    return chain