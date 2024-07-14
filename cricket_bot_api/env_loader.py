import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Accessing the variables
nvidia_api_key = os.getenv('NVIDIA_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')

print(nvidia_api_key)
print(cohere_api_key)
