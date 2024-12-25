import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

class CustomPythonTool:
    def __init__(self):
        # Initialize database connection with provided configuration
        self.conn = psycopg2.connect(
            dbname=os.getenv("db_name"),
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port")
        )
        self.cursor = self.conn.cursor()

    def run(self, query):
        try:
            # Execute the SQL query
            self.cursor.execute(query)
            if "UPDATE" in query:
                return "Updated successfully"
            # Fetch the result
            result = self.cursor.fetchall()
            # Retrieve column names
            column_names = [desc[0] for desc in self.cursor.description]
            # Combine column names with result rows
            result_with_columns = [dict(zip(column_names, row)) for row in result]
            if 'column_name' in query:
                formatted_result = [(f"lower({row[0]})",) for row in result]
                return formatted_result
            else:
                return result_with_columns
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            # Commit if there are any updates
            self.conn.commit()
python_tool = CustomPythonTool()
    
query ="UPDATE employee_details SET vacation_leaves_remaining = vacation_leaves_remaining - 1, pending_approval = 1 WHERE lower(name) = 'siva sai'"

result = python_tool.run(query)

print(result)
