import os
import re
from abc import abstractmethod

import pandas as pd
from openai import AzureOpenAI

from ..base import SqlCxCopilotBase


class AzureOpenAI_Chat(SqlCxCopilotBase):
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

    @staticmethod
    def system_message(message: str) -> dict:
        return {"role": "system", "content": message}

    @staticmethod
    def user_message(message: str) -> dict:
        return {"role": "user", "content": message}

    @staticmethod
    def assistant_message(message: str) -> dict:
        return {"role": "assistant", "content": message}

    @staticmethod
    def str_to_approx_token_count(string: str) -> int:
        return len(string) / 4

    @staticmethod
    def add_ddl_to_prompt(
        initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += f"\nYou may use the following DDL statements as a reference for what tables might be available. Use responses to past questions also to guide you:\n\n"

            for ddl in ddl_list:
                if (
                    AzureOpenAI_Chat.str_to_approx_token_count(initial_prompt)
                    + AzureOpenAI_Chat.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    @staticmethod
    def add_documentation_to_prompt(
        initial_prompt: str, documentation_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += f"\nYou may use the following documentation as a reference for what tables might be available. Use responses to past questions also to guide you:\n\n"

            for documentation in documentation_list:
                if (
                    AzureOpenAI_Chat.str_to_approx_token_count(initial_prompt)
                    + AzureOpenAI_Chat.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    @staticmethod
    def add_sql_to_prompt(
        initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += f"\nYou may use the following SQL statements as a reference for what tables might be available. Use responses to past questions also to guide you:\n\n"

            for question in sql_list:
                if (
                    AzureOpenAI_Chat.str_to_approx_token_count(initial_prompt)
                    + AzureOpenAI_Chat.str_to_approx_token_count(question["sql"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt

    @staticmethod
    def add_label_values_to_prompt(
        initial_prompt: str, label_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(label_list) > 0:
            initial_prompt += f"""\nThe question of the user is an approximate spelling of the valite proper noun the database. \
                You may to use the following proper nouns, labels or codes as a reference for how you can filter the data. \
                Use the noun, label or code most similar to the user original question. 
                Sometimes the user uses codified country names, if the user says US or USA this means united states.\
                The followiing nouns, codes etc, has the name of the column wich you can filter the data. \
                In the SQL query try to use the lowercase version of the proper noun, label or code, and apply LOWER to the corresponding column for example LOWER(Country). \
                Remember change the user input for the correct valid proper code, noun in the database for example, 'refractory hot spot' maybe would be 'refractory concrete' or something else instead the original user input. \
                Use responses to past questions also to guide you:\n\n"""

            for label in label_list:
                if (
                    AzureOpenAI_Chat.str_to_approx_token_count(initial_prompt)
                    + AzureOpenAI_Chat.str_to_approx_token_count(label)
                    < max_tokens
                ):
                    initial_prompt += f"{label}\n\n"
        return initial_prompt

    def get_sql_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        label_list: list,
        dialect: str = "delta table",
        **kwargs,
    ):
        initial_prompt = """You are an agent designed to interact with a SQL database. \
Given an input question, create a syntactically correct {dialect} sql query to run. \
You can order the results by a relevant column to return the most interesting examples in the database. \
Never query for all the columns from a specific table, only ask for the relevant columns given the question. \
You MUST double check your query. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\
The user provides a question and you provide SQL. You will only respond with SQL code and not with any explanations.\n\nRespond with only SQL code. Do not answer with any explanations -- just the code.\n"""
        initial_prompt = initial_prompt.format(dialect=dialect)
        initial_prompt = AzureOpenAI_Chat.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=14000
        )

        initial_prompt = AzureOpenAI_Chat.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=14000
        )

        initial_prompt = AzureOpenAI_Chat.add_label_values_to_prompt(
            initial_prompt=initial_prompt,
            label_list=label_list,
            max_tokens=14000,
        )

        message_log = [AzureOpenAI_Chat.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(
                        AzureOpenAI_Chat.user_message(example["question"])
                    )
                    message_log.append(
                        AzureOpenAI_Chat.assistant_message(example["sql"])
                    )

        message_log.append({"role": "user", "content": question})
        print(initial_prompt, "prompt")

        return message_log

    def get_followup_questions_prompt(
        self,
        question: str,
        df: pd.DataFrame,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = AzureOpenAI_Chat.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=14000
        )

        initial_prompt = AzureOpenAI_Chat.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=14000
        )

        initial_prompt = AzureOpenAI_Chat.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=14000
        )

        message_log = [AzureOpenAI_Chat.system_message(initial_prompt)]
        message_log.append(
            AzureOpenAI_Chat.user_message(
                "Generate a list of followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions."
            )
        )

        return message_log

    def generate_question(self, sql: str, **kwargs) -> str:

        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    def _extract_python_code(self, markdown_string: str) -> str:
        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    def submit_prompt(self, prompt, **kwargs) -> str:
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        # Count the number of tokens in the message log
        num_tokens = 0
        for message in prompt:
            num_tokens += (
                len(message["content"]) / 4
            )  # Use 4 as an approximation for the number of characters per token

        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL"),
            messages=prompt,
            max_tokens=500,
            temperature=0,
        )

        return response.choices[
            0
        ].message.content  # If no response with text is found, return the first response's content (which may be empty)


"""Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
