from abc import abstractmethod
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

from ..base import SqlCxCopilotBase

load_dotenv()


class AzureOpenAI_Embeddings(SqlCxCopilotBase):
    def __init__(self, client=None, config=None):
        SqlCxCopilotBase.__init__(self, config=config)

        if client is not None:
            self.client = client
            return

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def generate_embedding(self, data: str, **kwargs) -> list[float]:

        embedding = (
            self.client.embeddings.create(
                input=[data], model=os.getenv("AZURE_EMBEDDING_ENGINE")
            )
            .data[0]
            .embedding
        )

        return embedding
