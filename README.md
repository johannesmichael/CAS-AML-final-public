
# Chat with your Emails

The project tries to solve a common problem: searching for emails by context instead of email adresses or keywords only. It also lets you summarize the content of an email conversation. And for reference and more detailed information it provides you with the web link (if used on Outlook 365).

The project is the final project of the **CAS for Advanced Machine Learning 2022** at the University of Bern, Switzerland.






## Goals

- Create a real world application
- Learn about the preprocessing of the data and the time it takes until you can use them
- create a standard to compare with other approaches, like other models
- Create scripts, not notebooks, to make it usable for a pipeline
- get insights in to the fast changing world of LLMs and related libraries
- solve the dependency hell of python
- learn about multiprocessing and distributed compute
- stard developing a playground for internal use and future development
## Features

- Download Emails from your Outlook Inbox
- Create an Azure Blob Storage table with the content
- Get the latest Email by conversation
- Use OpenAI API to structure the content and create a summary
- Embedd that structured content with OpenAI Embedding API
- Create a vectorstor to store the embeddings
- Use Streamlit App to query your Emails


## Next steps / To Do

- Embedd content of attachments, so it can be queried as well
- add other vectorstores like Redis, FAISS, Pinecone, etc. to evaluate different stacks
- create own Retriever, to improve results
- use open source models to compare with OpneAI
- create playground to use different models and vectorstores side by side
- create pipeline to ingest new Emails regularly 

Follow up project:
- finetune models to see if results improve (started here https://github.com/johannesmichael/lit-gpt-local)
- try to create own model with company data to adress domain knowledge (based on the paper TinyStories https://arxiv.org/abs/2305.07759)
## Tech Stack

**Language:** Python 3.10

**Main packages:** Langchain, Chromadb, Streamlit, Poetry

**Databases:** Azure Blob Storage, Chromadb

**Models:** GPT-3.5-turbo, GPT-3.5-turbo-16k, GPT-4, text-embedding-ada-002


## Requirements

- For use of OpenAI API an API-Key is needed.
- For use of Microsofts Outlook REST API an Azure subscription. Details here: https://learn.microsoft.com/en-us/outlook/rest/get-started
- 
## Environment Variables

First, rename `environment.env` to `.env`

Add the following environment variables to your .env file or system



`OPENAI_API_KEY` -->  OpenAI API-Key

`OUTLOOK_CONTENT_CONNECTION_STRING`  --> Connection string for Azure Blob Storage


## Installation

To install the necessary dependencies, follow these steps:

1. Clone the repository to your local machine (only Linux Systems, I used WSL on Windows 11).
2. Navigate to the project directory.
3. The project uses poetry and conda for creating the environment (detailed instructions here: https://stackoverflow.com/questions/70851048/does-it-make-sense-to-use-conda-poetry)


```bash
conda create --name CAS --file conda-linux-64.lock
conda activate CAS
poetry install
```


For updating the environment add the conda packages to the environment.yml and use

    # Re-generate Conda lock file(s) based on environment.yml
    conda-lock -k explicit --conda mamba
    # Update Conda packages based on re-generated lock file
    mamba update --file conda-linux-64.lock
    # Update Poetry packages and re-generate poetry.lock
    poetry update

For adding packages from PyPi use

    poetry add <package_name==version>
## Usage/Examples

The project has the following structure:

```
.
├── Streamlit
│   ├── assets
│   └── pages
├── archiv
├── db
│   └── index
├── notebooks
└── scripts

8 directories

```

Detailed tree structure with files can be found in tree.txt

### scripts

Starting point of the project. The scripts are numbered in order of execution.

- 01_get_email_content.py: This script is used to retrieve the content of the emails.
- 02_prepare_email_gpt.py: This script uses OpenAI API to process the content.
- 03_openai_embeddings_chromadb.py: This script generates embeddings using the OpenAI model and stores the emails in the vectorstore.
- 04_query.py: This script is used to query the vectorstore.
- 04a_query_multiretriver.py: This script is uses the MultiRetriver from Langchain.

### db

Folder to store the vectorstore files

### Streamlit

Folder for the Streamlit App. 
Just use

    cd Streamlit
    streamlit run home.py
Output:

    You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
   



and click on the localhost URL provided to see the app.
Select Outlook from the Menue
![Streamlit](assets/streamlit.png)

![Streamlit](assets/streamlit2.png)


On the left side you can adjust the settings for

1. Model: Works best with gpt-3.5-turbo-16k because of the lenght of some emails
2. Collection: To experiment with the different chunksizes. 1000 seems to work best for longer emails
3. Temperature: Call it the creativity allowed by the model when generating answers. Default to 0, which works best for this use case.
4. Max. Tokens: Maximun number of tokens the answer generated by the model can have. Depends on the model chosen.
5. Number of documents to retrieve from the vectorstore.

The main window:

Query: Enter your instructions here
Answer: Answer generated by the model based on the query
Source Documents: The number of emails used for generating the query. Providing a weblink to the email in outlook365.


### notebooks

The collection of notebooks used to develop and test the scripts. Each notebook contains a header explaining the purpose. They are all left like they were after testing, so it is work in progress.

### archive

Collection of drafts made during the development process. For example attempts to run scripts with multiprocessing, asyncio or distributed with Ray or Dask.

### singele files in root folder

Files for setting up the conda environment

    conda-linux-64.lock
    conda-lock.yml
    environment.yml
    poetry.lock
    pyproject.toml

Copy of the `.env` file if present in .gitignore. Just rename it and adjust to your needs, like adding the environment variables

    environment.env

Configuration for use in debug mode in VS Code

    launch.json




