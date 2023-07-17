#This is the module from langchain, that checks the embeddings.
#Since the implemented chromadb was not woking because of empty arrays, I tried to fix it here
#but decided to use chromadb directly instead.




def _get_len_safe_embeddings(
    self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
) -> List[List[float]]:
    embeddings: List[List[float]] = [[] for _ in range(len(texts))]
    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "Could not import tiktoken python package. "
            "This is needed in order to for OpenAIEmbeddings. "
            "Please install it with `pip install tiktoken`."
        )

        

    tokens = []
    indices = []
    encoding = tiktoken.model.encoding_for_model(self.model)
    for i, text in enumerate(texts):
        if self.model.endswith("001"):
            # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
            # replace newlines, which can negatively affect performance.
            text = text.replace("\n", " ")
        token = encoding.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )
        for j in range(0, len(token), self.embedding_ctx_length):
            tokens += [token[j : j + self.embedding_ctx_length]]
            indices += [i]

    batched_embeddings = []
    _chunk_size = chunk_size or self.chunk_size
    for i in range(0, len(tokens), _chunk_size):
        print(f"Embedding chunk {i} to {i + _chunk_size}.")
        first_response = embed_with_retry(
            self,
            input=tokens[i : i + _chunk_size],
            **self._invocation_params,
        ) 
        to_retry = [(tokens[i], i) for i in range(len(first_response["data"])) if len(first_response["data"][i]["embedding"]) < 1536]
        print(f"Found {len(to_retry)} errors.")
        attempt = 0
        while len(to_retry) > 0:
            attempt += 1
            cur_token, idx = to_retry[-1]
            response = embed_with_retry(
                self,
                input=[cur_token],
                **self._invocation_params,
            )
            if len(response["data"][0]["embedding"]) == 1536:
                print(f"Replacing embedding for chunk {idx}.")
                first_response["data"][idx]["embedding"] = response["data"][0]["embedding"]
                to_retry.pop()
        batched_embeddings += [r["embedding"] for r in first_response["data"]]
        print(f"Finished embedding chunk {i} to {i + _chunk_size} in {attempt} attempts.")

    print("Finished embedding.")

    results: List[List[List[float]]] = [[] for _ in range(len(texts))]
    num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
    for i in range(len(indices)):
        results[indices[i]].append(batched_embeddings[i])
        num_tokens_in_batch[indices[i]].append(len(tokens[i]))

    for i in range(len(texts)):
        _result = results[i]
        if len(_result) == 0:
            average = embed_with_retry(
                self,
                input="",
                **self._invocation_params,
            )[
                "data"
            ][0]["embedding"]
        else:
            average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
        embeddings[i] = (average / np.linalg.norm(average)).tolist()

    return embeddings