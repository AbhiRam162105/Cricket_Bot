# Cricket Chatbot with LangChain and FastAPI

This project demonstrates how to build a cricket-themed chatbot using LangChain for natural language understanding and FastAPI for creating a web API. The chatbot can answer questions related to cricket by processing and analyzing documents from Wikipedia, generating embeddings, and utilizing NVIDIA AI models for generating responses. For more elaborate explanation of the code refer to my story on Medium
-  https://medium.com/@abhiyanampally/simplest-introduction-to-rag-and-langchain-building-a-cricket-chatbot-part-i-0b98a658ee6f
-  https://medium.com/@abhiyanampally/simplest-introduction-to-rag-and-langchain-building-a-cricket-chatbot-part-2-308b29ce1436


![Cricket Chatbot UI](https://github.com/user-attachments/assets/1c3b527c-755f-4c36-9730-729ba51dbcc3 "Cricket Chatbot User Interface")




## Features

-   Employs Retrieval-Augmented Generation (RAG) to combine the strengths of retrieval and generative models, enhancing the accuracy of chatbot responses.
-   Utilizes LangChain, a comprehensive framework designed to simplify the development of advanced natural language processing models like RAG.
-   Processes and analyzes cricket-related documents sourced from Wikipedia, providing a rich knowledge base for the chatbot.
-   Creates document chunks and leverages vector embeddings for efficient information retrieval, enabling the chatbot to access relevant content quickly.
-   Implements a self-querying retriever mechanism within LangChain, allowing the chatbot to dynamically query its knowledge base based on user inputs.
-   Integrates chat memory to maintain context across interactions, improving the coherence and relevance of responses over time.
-   Configures the chatbot to generate human-like responses using NVIDIA NIM Inference models, ensuring answers are both informative and engaging.
-   Provides a RESTful API using FastAPI, facilitating easy integration of the chatbot into various applications and platforms.
-   Offers a comprehensive guide within a Jupyter Notebook environment, making it accessible for users to follow along and build their own cricket chatbot.
- A user-friendly ReactJS frontend to facilitate intuitive interaction with the cricket chatbot, offering a seamless user experience.


## Requirements

- Python 3.7+

- FastAPI

- Uvicorn (for running the FastAPI app)

- LangChain (and its dependencies)

- NVIDIA AI model access (requires API keys)

## Setup

1\. Clone the repository:

```bash

git clone https://github.com/yourusername/cricket-chatbot.git

cd cricket-chatbot

```

2\. Install the required packages:

```bash

pip install -r requirements.txt

```

3\. Set up environment variables for NVIDIA and Cohere API keys:

```bash

export NVIDIA_API_KEY=your_nvidia_api_key_here

export COHERE_API_KEY=your_cohere_api_key_here

```

## Running the Application

To start the FastAPI server, run:

```bash

uvicorn cricket_bot_api:app --reload

```

This will start the server at `http://localhost:8000`. You can interact with the chatbot through the `/chat` endpoint by sending POST requests with JSON payloads containing the user's question.

Example request body:

```json

{

  "text": "Who won the last Cricket World Cup?"

}

```

# Setting Up a ReactJS Frontend for the Cricket Chatbot Project

To complement the backend capabilities of the Cricket Chatbot project, a ReactJS frontend can be developed to provide a user-friendly interface for interacting with the chatbot. This guide outlines the steps to set up the ReactJS frontend, assuming you have Node.js and npm installed on your system. It also assumes that the FastAPI server is running on `http://localhost:8000`.

## Frontend User Interface

The frontend UI will serve as the primary interface through which users can interact with the cricket chatbot. It will display the conversation history and provide an input field for users to type their questions.

## Step 1: Clone the Repository

First, clone the repository containing the frontend code to your local machine. Open a terminal and execute the following commands:

```bash

git clone https://github.com/AbhiRam162105/Cricket_Bot.git

cd Cricket_Bot/frontend

```

This will download the project files and navigate into the `frontend` directory.

## Step 2: Install Dependencies

Before starting the project, it's necessary to install all dependencies listed in the `package.json` file. Run the following command in the terminal:

```bash

nvm install 16.13.2

npm install

```

This command ensures that Node.js version `16.13.2` is installed and then reads the `package.json` file to install all the required packages.

## Step 3: Start the Development Server

Once the installation is complete, you can start the development server to see the application in action. Execute the following command:

```bash

npm start

```

This command compiles the React app and starts a development server. By default, the app will be served at `http://localhost:3000`. You can now open this URL in your web browser to interact with the cricket chatbot.

## Note

Ensure that the FastAPI backend server is running on `http://localhost:8000` as the ReactJS frontend communicates with this backend to fetch chatbot responses. If the backend server is hosted elsewhere, you may need to update the API endpoint URLs in the frontend code accordingly.

By following these steps, you will have successfully set up a ReactJS frontend for the Cricket Chatbot project, providing a seamless user experience for interacting with the chatbot.


## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.




