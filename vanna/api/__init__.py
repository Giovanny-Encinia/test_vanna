import json
from typing import Dict, List, Union


def post(endpoint: str = "", headers: Dict = {}, data: Dict = {}) -> Dict:
    match data.get("method"):
        case "add_user_to_org":
            result = {
                "jsonrpc": "2.0",
                "result": {"message": "User added to organization", "success": True},
            }
            return result
        case "list_orgs":
            result = {
                "jsonrpc": "2.0",
                "result": {"organizations": ["demo-tpc-h", "test-org", "test-org2"]},
            }
            return result
        case "create_org":
            result = {
                "jsonrpc": "2.0",
                "result": {"message": "Organization created", "success": True},
            }
            return result
        case "set_org_visibility_update_model_visibility":
            result = {
                "error": {
                    "code": -32603,
                    "message": "KeyError('set_org_visibility_update_model_visibility')",
                },
                "jsonrpc": "2.0",
            }
            return result
        case "store_sql":
            # this case is for the question store, store the question
            # in redis and return the result
            """
                    vn.add_sql(
                question="What is the average salary of employees?",
                sql="SELECT AVG(salary) FROM employees"
            )
                    @dataclass
                    class QuestionSQLPair:
                        question: str
                        sql: str
                        tag: Union[str, None]
            """
            data = {
                "method": "store_sql",
                "params": [
                    {
                        "question": "Which 10 domains received the highest amount of traffic on Black Friday in 2021 vs 2020",
                        "sql": "SELECT domain,\n       sum(case when date = '2021-11-26' then total_visits\n                else 0 end) as visits_2021,\n       sum(case when date = '2020-11-27' then total_visits\n                else 0 end) as visits_2020\nFROM   s__p_500_by_domain_and_aggregated_by_tickers_sample.datafeeds.sp_500\nWHERE  date in ('2021-11-26', '2020-11-27')\nGROUP BY domain\nORDER BY (visits_2021 - visits_2020) desc limit 10",
                        "tag": "Manually Trained",
                    }
                ],
            }

            result = {
                "jsonrpc": "2.0",
                "result": {"message": "Successfully stored question", "success": True},
            }
            return result
        case "store_ddl":
            """vn.add_ddl(
                ddl="CREATE TABLE employees (id INT, name VARCHAR(255), salary INT)"
            )"""
            result = {
                "jsonrpc": "2.0",
                "result": {"message": "Successfully stored DDL", "success": True},
            }
            return result
        case "store_documentation":
            """vn.add_documentation(
                documentation="Our organization's definition of sales is the discount price of an item multiplied by the quantity sold."
            )"""

            result = {
                "jsonrpc": "2.0",
                "result": {
                    "message": "Successfully stored documentation",
                    "success": True,
                },
            }
            return result
        case "set_accuracy_category":
            result = {
                "jsonrpc": "2.0",
                "result": {
                    "message": "Failed to set the accuracy category: (sqlite3.IntegrityError) NOT NULL constraint failed: answer_tag.answer_id\n[SQL: INSERT INTO answer_tag (answer_id, tag) VALUES (?, ?)]\n[parameters: (None, 'Flagged for Review')]\n(Background on this error at: https://sqlalche.me/e/20/gkpj)",
                    "success": False,
                },
            }

            return result
        case "remove_sql":
            """vn.remove_sql(question="What is the average salary of employees?")"""
            result = {
                "jsonrpc": "2.0",
                "result": {"message": "Successfully removed question", "success": true},
            }
            return result
        case "generate_sql_from_question":
            result = {
                "jsonrpc": "2.0",
                "result": {
                    "postfix": "",
                    "prefix": "",
                    "raw_answer": "AI Response",
                    "sql": "No SELECT statement could be found in the SQL code",
                },
            }
            return result
        case "get_related_training_data":
            result = {
                "jsonrpc": "2.0",
                "result": {
                    "ddl": ["DDL here"],
                    "documentation": ["Documentation here"],
                    "questions": [
                        {
                            "question": "What is the total sales for each product?",
                            "sql": "SELECT * FROM ...",
                            "tag": None,
                        }
                    ],
                },
            }
            return result
        case "generate_meta_from_question":
            """vn.generate_meta(question="What tables are in the database?")
            # Information about the tables in the database"""
            result = {"jsonrpc": "2.0", "result": {"data": "AI Response"}}
            return result
        case "generate_followup_questions":
            result = {"jsonrpc": "2.0", "result": {"questions": ["AI Response"]}}
            return result
        case "generate_questions":
            """vn.generate_questions()
            # ['What is the average salary of employees?', 'What is the total salary of employees?', ...]
            """
            # this method if for generating questions and save in redis
            result = {"jsonrpc": "2.0", "result": {"questions": ["AI Response"]}}
            return result
        case "generate_plotly_code":
            """question (str): The question to generate Plotly code for.
            sql (str): The SQL query to generate Plotly code for.
            df (pd.DataFrame): The dataframe to generate Plotly code for.
            chart_instructions (str): Optional instructions for how to plot the chart.

            vn.generate_plotly_code(
                    question="What is the average salary of employees?",
                    sql="SELECT AVG(salary) FROM employees",
                    df=df
                )
                # fig = px.bar(df, x="name", y="salary")"""

            result = {"jsonrpc": "2.0", "result": {"plotly_code": "AI Response"}}
            return result
        case "generate_explanation":
            """vn.generate_explanation(sql="SELECT * FROM students WHERE name = 'John Doe'")
            # 'This query selects all columns from the students table where the name is John Doe.'
            """
            result = {"jsonrpc": "2.0", "result": {"plotly_code": "AI Response"}}
            return result
        case "generate_question":
            """vn.generate_question(sql="SELECT * FROM students WHERE name = 'John Doe'")
            # 'What is the name of the student?'"""
            result = {"jsonrpc": "2.0", "result": {"plotly_code": "AI Response"}}
            return result
        case "get_all_questions":
            result = {
                "jsonrpc": "2.0",
                "result": {
                    "data": '[{"question_id":"5c58036e55cbd5c73bb6c8f9c00edde0","created_at":"2024-02-09 16:44:14","question":"Who are the top 10 customers by Sales?","answer":"No SELECT statement could be found in the SQL code","tag":"Vanna Generated"}]'
                },
            }
            return result
        case "get_training_data":
            """Get the training data for the current model"""
            result = {"jsonrpc": "2.0", "result": {"data": "[]"}}
            return result
