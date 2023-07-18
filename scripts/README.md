
# Documentation


Detailed explanation of the scripts

## scripts



### 01_get_email_content.py

This script is used to fetch emails from a specific folder or subfolder in a Microsoft Outlook account within a specified date range, and then store the email content in an Azure Table Storage. 

It uses the Microsoft Authentication Library (MSAL) to authenticate and get an access token. It then uses the Microsoft Graph API to fetch the emails. 

A class `EmailMime` is used to parse the email content and extract the sender, recipients, subject, and content of the email. The content is cleaned to remove URLs, email headers, email addresses, addresses, and phone numbers. 

The `logging` module is used to log successful and failed operations, so one can check if all necessary emails are processed. 

The script is designed to be run from the command line with arguments for the start date, end date, and subfolder name. 


The most important functions and classes in the script are:

1. `get_token()`: This function is used to get an access token for authentication using the MSAL library.

2. `get_folder_id(folder_name, access_token, user_id)`: This function is used to get the folder ID of a specific folder by its name.

3. `get_subfolder_id(subfolder_name, access_token, user_id, folder_id)`: This function is used to get the subfolder ID of a specific subfolder by its name.

4. `get_email_ids_by_daterange(start_time, end_time, access_token, user_id, folder_id, subfolder_id)`: This function is used to get the email IDs within a specified date range from a specific folder or subfolder.

5. `EmailMime` class: This class represents an email and provides methods to fetch the MIME content of the email, parse the message, extract sender and recipient information, extract subject and content of the email, clean the content by removing URLs and sensitive information, convert the email data into a dictionary format, and save the email data to Azure Table Storage.

6. `AzureTable` class: This class represents an Azure Table Storage and provides methods to create an entity in the table.

7. `parse_args()`: This function is used to parse command line arguments for start date, end date, and subfolder name.

8. `main()`: This function is the main entry point of the script. It retrieves command line arguments using `parse_args()`, gets access token using `get_token()`, fetches folder and subfolder IDs using `get_folder_id()` and `get_subfolder_id()`, retrieves email IDs using `get_email_ids_by_daterange()`, processes each email using `EmailMime` class methods, saves each email to Azure Table Storage using `save_to_azure()`, and logs successful and failed operations.

### Usage

    cd scripts
    python 01_get_email_content.py --start_date "01-07-2022" --end_date "31-07-2022"
### 02_prepare_email_gpt.py



Note: The script uses OpenAI's API. Make sure you have set up your OpenAI API credentials before running the script.

This script is used to process and analyze email conversations stored in an Azure Table Storage. There are 3 main tasks:


1. `clean_data(df)`: This function cleans the DataFrame by dropping rows with empty content, keeping only the latest email in each conversation, cleaning the content of each email, decoding the subject, and calculating additional columns such as content length and token length.
By keeping only the last email of an conversation a lot of duplicated content can be removed, because it contains all prior emails.


2. `get_completion(value)`: This function queries OpenAI's GPT-3 model to analyze an email conversation given its content and returns information about senders, time, subjects, messages, and types.
This way, a lot of unnecessary content can be removed and the different parts of the conversation can be classified into these types: Question, Answer, Information, Advertising.

```
prompt = f"""
                Analysiere folgende Email-Unterhaltung, getrennt durch <>, nach folgenden Kriterien:
                - Sender
                - Gesendet
                - Betreff
                - Nachricht (nur Text, entferne Signaturen, Adressen, Bilder,
                   Links, Disclaimer und Fussnoten)
                - Typ (Frage, Antwort, Information, Aufforderung, Werbung...)

                Antwort in einer Liste. Einträge getrennt durch <br>. Beispiel:
                
                <br>
                Typ: 
                Sender: 
                Gesendet: 
                Subject: 
                Nachricht:
                <br>
                <{value}>
                """

```

3. `get_summary(value)`: Used to generate a summary of an email conversation given its content.
The model to be used is dependend on the token count. above 3000 tokens it uses the `gpt-3.5-turbo-16k` model, which has a greater context length.

Prompt:
 ```
prompt = f"""Erstelle eine Zusammenfassung der folgenden Email-Unterhaltung in <>,
   inklusive der Personen, die daran beteiligt sind.
                Beispiel:
                Personen: 
                Zusammenfassung: 
                <{value}>
                 """
```

At the end, it uploads the dataframe as a parquet file to Azure Blob Storage for further use in the following scripts.

### 03_openai_embeddings_chromadb.py

The script is designed to process and analyze email data from an Azure Blob Storage. It downloads the data, cleans it, and prepares it for embedding. The script then creates embeddings for the text data using the OpenAI API. The embeddings are then uploaded back to the Azure Blob Storage. The script also checks if a ChromaDB vectorstore exists and either appends to the existing vectorstore or creates a new one. The processed data is then added to a collection for further use. 


1. `get_data(file_name)`: This function retrieves data from Azure Blob Storage using the BlobServiceClient. It takes a file name as input and returns a dataframe.

2. `create_list(value)`: This function creates a list of dictionaries from the JSON in the 'content_processed' column of the dataframe.

3. `create_text(value)`: This function creates a text from the content keys in the JSON.

4. `clean_data(file_name)`: This function cleans the data and prepares it for embedding. It drops certain rows, creates new columns, and removes unnecessary columns.

5. `get_embedding(content, model="text-embedding-ada-002")`: This function creates embeddings for the text using the OpenAI API.
Here the use of multiprocessing is used, since the Embedd API is very fast and the rate limit is high.

```
    # Using multiprocessing with 4 processes
    with Pool(4) as p:
        embeddings = list(tqdm(p.imap(request_embeddings, texts),
         total=len(texts), desc="Creating embeddings"))
    # Extract metadata
    metadatas = [doc.metadata for doc in texts]

    # Create DataFrame
    df_embeddings = DataFrame(metadatas)
    df_embeddings['embedding'] = embeddings
    #create id column
    df_embeddings['uuid'] = [str(uuid.uuid4()) for _ in range(len(df_embeddings.index))]

```



6. `process_data(df, chunk_size, chunk_overlap)`: Splitting content into chunks of text. Uses the `RecursiveCharacterTextSplitter` from Langchain.

```

    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")



```

7. `request_embeddings(doc)`: This function requests embeddings for a document, using the `text-embedding-ada-002` model from OpenAI. It is used in `create_embeddings()` to allow multiprocessing. 

8. `create_embeddings(texts)`: Creates embeddings for a list of texts using multiprocessing.

9. `does_vectorstore_exist(persist_directory: str)`: This function checks if a vectorstore exists.


10. `main()`: After retrieving and cleaning the data, we check if the vectorstore already exist. IF not, a new one is created. If yes, it checks if the collection we use in that script already exist and creates a new one if `False`. (If `True` case is not handled right now, so only new collections can be used.)
Then all the embeddings are added to that collection with the appropriate metadata.

```
#function to add to collection
def add_to_collection(row):
    collection.add(documents=row['content'],
    embeddings=row['embedding']['data'][0]['embedding'],
     metadatas=row[['subject', 'conversation_id', 'web_link', 'display_text']].to_dict(),
    ids=[str(row['uuid'])])
    return True 
    
#add to collection
df_embeddings.progress_apply(add_to_collection, axis=1)

```
The metadata is later used to display content and web_link in the query.


The script also uses environment variables to get connection strings and API keys, and uses the argparse library to parse command-line arguments. 

To run the script: 

    python 03_openai_ada_embeddings.py \ 
    --file_name 2021-06-30_outlook_ada_embeddings_csX.parquet \
    --collection_name openai_ada_1000cs \
    --chunk_size 1000 --chunk_overlap 50

Replace the file_name with the file_name from the previous script.
### 04_query.py

This script is a chatbot application that uses OpenAI's GPT-3 model for generating responses to user queries. It also uses a retrieval-based question answering system, which retrieves information from a database to answer user queries.

It retrieves the directory for persisting data and the OpenAI API key from the environment variables. It initializes an OpenAI embedding function with the specified model name.

The `parse_args()` function is used to parse command-line arguments. It creates an argument parser and adds two arguments: `collection_name` and `max_tokens`. The `collection_name` argument is used to specify the name of the collection to  use, and `max_tokens` is used to specify the maximum number of tokens to generate.

In the main part of the script, it first parses the command-line arguments and retrieves the values of `collection_name` and `max_tokens`. It then creates a `Chroma` object, which is a database for storing and retrieving embeddings. The `Chroma` object is initialized with the collection name, persist directory, embedding function, and client settings.

Next, it creates a retriever from the `Chroma` database and a language model using OpenAI's GPT-3 model. It then creates a retrieval-based question answering system using the language model and the retriever.

Finally, it enters a loop where it continuously takes user input, generates a response using the question answering system, and prints the response. The loop continues indefinitely until the script is stopped.

To run the script:

    python 04_query.py --collection_name openai_ada_1000cs --max_tokens 1000
### 04a_query_multiretriver.py

This script is designed to retrieve relevant documents from a vector database using multiple versions of a user's question. It uses OpenAI's language model and the Chroma library for document retrieval.

The `load_dotenv()` function loads environment variables from a .env file into the shell’s environment variables so they can be used in the script.

`CHROMA_SETTINGS`, `PERSIST_DIRECTORY`, and `OPENAI_API_KEY` are constants imported or retrieved from the environment, which are used later in setting up the Chroma database and OpenAI API key respectively.

The following classes are from the `langchain` library.

The `LineList` class is a Pydantic model that represents lines of text as list of strings. The `LineListOutputParser` class inherits from `PydanticOutputParser- , it overrides its parse method to split input text by newline characters ("\n") and return an instance of `LineList` with these lines.

The `QUERY_PROMPT` constant defines how questions will be formatted when sent to OpenAI's language model (`gpt-3.5-turbo-16k`).


```
template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions seperated by newlines.
    Original question: {question}"""

```

In main execution block (`if __name__ == "__main__":`), command line arguments are parsed using argparse module to get collection name and maximum tokens allowed for generation. 

An instance of ChatOpenAI is created with specified parameters like temperature, model_name, max_tokens etc., this object serves as interface for interacting with  models provided by OpenAi. This llm object along with `QUERY_PROMPT` & `output_parser` objects are passed while creating `LLMChain` object which helps in chaining together different components involved in generating queries based on user input.

Next, an embeddings object is created using `OpenAIEmbeddings()` function call which would help us generate embeddings for our data points/documents.
 
Then we create an instance of Chroma DB where we pass collection_name (to specify which collection/database to use), persist_directory (where data is stored), embedding_function (how embeddings should be generated) & client_settings related information about chroma db setup details.
  
Finally `MultiQueryRetriever` object is instantiated where `retriever=db.as_retriever()`, `llm_chain=llm_chain` & `parser_key="lines"` are passed as arguments. This retriever acts as bridge between query generator(llm_chain) & document fetcher(retriever). 

In infinite loop at end, user inputs their query/question via console/terminal; this query gets processed through `MultiQueryRetriever` pipeline resulting in fetching most relevant documents matching against given query; those results then get printed out onto console/terminal screen.
## Streamlit

The Streamlit app is used to provide a nice interface for querying the vectorstore.
It consitst of two scripts, one for the Homepage and one for the actual query interface.
### 01_Outlook.py

This script is a Streamlit application that uses OpenAI's GPT models to answer questions about the user's emails. The email content is stored in a vectorstore database and queried using the Chroma library, which retrieves the most relevant documents for a given query.

The main functions of this script are:

1. `st.set_page_config()`: This function sets up the page configuration including title, icon, layout and initial sidebar state.

2. `st.sidebar.radio()`: This function creates radio buttons on the sidebar for users to select models or chunksize options.

3. `st.sidebar.slider()`: These sliders allow users to adjust parameters like `temperature` (which controls creativity of model), `max_tokens` (length of answer) and `number_docs` (number of documents) retrieved from database.

4. `parse_arguments()`: This function parses command line arguments such as whether to hide source documents used for answers or mute streaming StdOut callback for LLMs.

5. `query_chat()` : It takes in several parameters including query string, number of docs, max tokens etc., then it initializes embeddings with OpenAI API key, sets up retriever with Chroma settings and finally gets an answer from chain using RetrievalQA class from ChatOpenAI module.



```
def query_chat(query_string, num_docs=3, max_tokens=2000, temperature=0.5, model_name="gpt-3.5-turbo"):

    # Parse the command line arguments
    args = parse_arguments()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(collection_name=collection_name, persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens, callbacks=callbacks, verbose=False, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    
    # Get the answer from the chain
    res = qa(query_string)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents'][0:num_docs]
    
    return answer, docs


```

6.`extract_doc()` : It extracts page content and metadata from each document returned by chatbot response


7.`if execute:`: When 'Execute Query' button is clicked on UI , it runs `query_chat` method with provided inputs & displays output on screen along with source links if available.
   
These functions work together by taking user input through Streamlit interface elements, processing these inputs via OpenAI API calls & displaying results back onto Streamlit app interface .