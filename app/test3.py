
import os
from langchain.agents import create_openai_tools_agent, Tool
from tools import initialize_tools
from dotenv import load_dotenv
from langchain import hub
from custom_tools import CustomPythonTool
_ = load_dotenv()

python_tool = CustomPythonTool()
tools = [Tool(
            name="Employee Data",
            func=python_tool.run,
            description = f"""
            Useful for when you need to answer questions about employee data stored in postgres sql dataset. 
            Run postgres sql query operations on dataset employee_details to help you get the right answer.
            
            Note: Whenever there is a where clause in query definitely user lower(column_name)
            Example: SELECT name FROM employee_details WHERE LOWER(employment_status) = 'probation'

            Below is the example conversation.
            <user>: How many Sick Leave do I have left?
            <assistant>
                Thought: I need to check the employee data to get this information. Convert field to lower case while using where clause to make it case insensitive.
                Action: Employee Data
                Action Input: Select sick_leaves_remaining From employee_details Where LOWER(name) = 'siva sai'
                Observation: 45
                Thought: I now know the final answer.
                Final Answer: You have 45 sick leaves left.
            </assistant>

            <user>: Please book 1(one) vacation leave for me.
            <assistant>
                Thought: I will deduct one leave from the remaining balance and add one leave to pending_approval and update the database.
                Action: Employee Data
                Action Input: UPDATE employee_details SET vacation_leaves_remaining = vacation_leaves_remaining - 1, pending_approval = pending_approval + 1 WHERE lower(name) = 'siva sai';
                Observation: Leave balance updated successfully in the database.
                Final Answer: I booked one vacation leave for you and it is pending manager approval.
            </assistant>
            """
        )]
user = 'Siva Sai'
prompt = hub.pull("hwchase17/openai-functions-agent")

to, llm = initialize_tools(api_key=os.getenv("GROQ_API_KEY"))



inputs = {
    "input": "Hi",
    "intermediate_steps": []
}
#agent_out = query_agent_runnable.invoke(inputs)
agent = llm.bind_tools(tools, tool_choice = "Employee Data")
agent_out = agent.invoke("How many sick leaves I have?")
print(agent_out)