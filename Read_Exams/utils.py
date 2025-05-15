# Note: code written with the help of Open AI' ChatGPT o3 model
import os
from dotenv import load_dotenv

def load_api_key():
    """"Initialization function used to get API key from .env file"""
    load_dotenv()   # Load variables from .env into environment
    key = os.getenv("OPENAI_API_KEY")
    if key is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return key

    


