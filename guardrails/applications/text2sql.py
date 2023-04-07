from typing import Callable, Dict, Optional

import openai

from guardrails.document_store import DocumentStoreBase, EphemeralDocumentStore
from guardrails.embedding import EmbeddingBase, OpenAIEmbedding
from guardrails.guard import Guard
from guardrails.utils.sql_utils import create_sql_driver
from guardrails.vectordb import Faiss, VectorDBBase

REASK_PROMPT = """
You are a data scientist whose job is to write SQL queries.

@complete_json_suffix_v2

Here's schema about the database that you can use to generate the SQL query.
Try to avoid using joins if the data can be retrieved from the same table.

{{db_info}}

I will give you a list of examples.

{{examples}}

I want to create a query for the following instruction:

{{nl_instruction}}

For this instruction, I was given the following JSON, which has some incorrect values.

{previous_response}

Help me correct the incorrect values based on the given error messages.
"""


def example_formatter(input: str, output: str, output_schema: Callable) -> str:
    if output_schema is None:
        output = output_schema(output)

    example = "\nINSTRUCTIONS:\n============\n"
    example += f"{input}\n\n"

    example += "SQL QUERY:\n================\n"
    example += f"{output}\n\n"

    return example


class Text2Sql:
    def __init__(
        self,
        conn_str: str,
        schema_file: Optional[str] = None,
        examples: Optional[Dict] = None,
        embedding: Optional[EmbeddingBase] = OpenAIEmbedding,
        vector_db: Optional[VectorDBBase] = Faiss,
        document_store: Optional[DocumentStoreBase] = EphemeralDocumentStore,
        rail_spec: Optional[str] = "text2sql.rail",
        example_formatter: Optional[Callable] = example_formatter,
        reask_prompt: Optional[str] = REASK_PROMPT,
    ):
        """Initialize the text2sql application.

        Args:
            conn_str: Connection string to the database.
            schema_file: Path to the schema file. Defaults to None.
            examples: Examples to add to the document store. Defaults to None.
            embedding: Embedding to use for the document store. Defaults to OpenAIEmbedding.
            vector_db: Vector database to use for the document store. Defaults to Faiss.
            document_store: Document store to use. Defaults to EphemeralDocumentStore.
            rail_spec: Path to the rail specification. Defaults to "text2sql.rail".
            example_formatter: Function to format examples. Defaults to example_formatter.
            reask_prompt: Prompt to use for reasking. Defaults to REASK_PROMPT.
        """

        self.example_formatter = example_formatter

        # Initialize the SQL driver.
        self.sql_driver = create_sql_driver(conn=conn_str, schema_file=schema_file)
        self.sql_schema = self.sql_driver.get_schema()

        # Initialize the document store.
        self.store = self._add_examples_to_docstore(
            examples, embedding, vector_db, document_store
        )

        # Initialize the Guard class
        self.guard = Guard.from_rail(rail_spec)
        self.guard.reask_prompt = reask_prompt

    def _add_examples_to_docstore(
        self,
        examples: Dict,
        embedding: EmbeddingBase,
        vector_db: VectorDBBase,
        document_store: DocumentStoreBase,
    ) -> EphemeralDocumentStore:
        """Add examples to the document store."""
        e = embedding()
        if vector_db == Faiss:
            db = Faiss.new_flat_l2_index(e.output_dim)
        else:
            raise NotImplementedError(f"VectorDB {vector_db} is not implemented.")
        store = document_store(db, e)
        store.add_texts(
            {example["question"]: {"ctx": example["query"]} for example in examples}
        )
        return store

    def __call__(self, text: str) -> str:
        """Run text2sql on a text query and return the SQL query."""

        similar_examples = self.store.search(text, 1)
        similar_examples_prompt = "\n".join(
            self.example_formatter(example.text, example.metadata["ctx"])
            for example in similar_examples
        )
        return self.guard(
            openai.Completion.create,
            prompt_params={
                "nl_instruction": text,
                "examples": similar_examples_prompt,
                "db_info": str(self.sql_schema),
            },
            engine="text-davinci-003",
            max_tokens=512,
        )
