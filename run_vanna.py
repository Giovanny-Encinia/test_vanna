from src.vanna.redis.redis_vectorstore import RedisVectorStore
from src.vanna.azureopenai.azureopenai_chat import AzureOpenAI_Chat
import yaml


class SQLCopilot(RedisVectorStore, AzureOpenAI_Chat):
    def __init__(self, config=None):
        RedisVectorStore.__init__(self)
        AzureOpenAI_Chat.__init__(self)


# with open("src/data/rca_sql_ddl.yml", "r") as ymlfile:
#     data = yaml.load(ymlfile, Loader=yaml.FullLoader)

copilot = SQLCopilot()
# copilot.train(ddl=data["ddl"])

print("Responding....")
copilot.connect_to_databricks_catalog()
result = copilot.ask(
    "what are the most common failure modes stoppages in United States?"
)
