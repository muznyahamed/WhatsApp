import json
import os
from typing import Dict, List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import JSONLoader
from dotenv import load_dotenv

# Load environment variables from .env file
# Try to load from current directory first, then parent directory
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    load_dotenv(dotenv_path='../.env')

class MobileStoreBot:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o",
            openai_api_key=openai_api_key
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        # Try to load data from both potential paths
        data_paths = ['data/', '../data/']
        
        # Load FAQs
        faq_loaded = False
        for path in data_paths:
            try:
                with open(f'{path}faq.json', 'r') as f:
                    self.faqs = json.load(f)['faqs']
                    faq_loaded = True
                    break
            except FileNotFoundError:
                continue
        
        if not faq_loaded:
            raise FileNotFoundError("Could not find faq.json in any of the data paths")
            
        # Load product data
        products_loaded = False
        for path in data_paths:
            try:
                with open(f'{path}products.json', 'r') as f:
                    self.products = json.load(f)
                    products_loaded = True
                    break
            except FileNotFoundError:
                continue
                
        if not products_loaded:
            raise FileNotFoundError("Could not find products.json in any of the data paths")
            
        # Create vector store from product data
        documents = []
        for phone in self.products['phones']:
            documents.append(str(phone))
        for acc in self.products['accessories']:
            documents.append(str(acc))
            
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vector_store = FAISS.from_texts(
            documents,
            embeddings
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory
        )
        
    def check_faq(self, query: str) -> Optional[str]:
        """Check if query matches any FAQ"""
        for faq in self.faqs:
            if query.lower() in faq['question'].lower():
                return faq['answer']
        return None
    
    def get_response(self, query: str) -> str:
        # First check FAQs
        faq_response = self.check_faq(query)
        if faq_response:
            return faq_response
            
        # If not FAQ, use RAG
        response = self.qa_chain({"question": query})
        return response['answer']

def main():
    # Get API key directly if not found from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        api_key = input("Please enter your OpenAI API key: ")
    
    # Initialize bot
    bot = MobileStoreBot(openai_api_key=api_key)
    
    print("Mobile Store Bot: Hello! How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Mobile Store Bot: Goodbye!")
            break
            
        response = bot.get_response(user_input)
        print(f"Mobile Store Bot: {response}")

if __name__ == "__main__":
    main() 