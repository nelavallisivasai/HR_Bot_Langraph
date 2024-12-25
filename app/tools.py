from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from llm_init import initialize_llm, load_embedding_model, initialize_vector_store
from custom_tools import CustomPythonTool, initialize_calculator

# Initialize tools
def initialize_tools(api_key):
    embedding_model = load_embedding_model()
    vector_store = initialize_vector_store(embedding_model)
    llm = initialize_llm(api_key)
    
    timekeeping_policy = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    python_tool = CustomPythonTool()
    calculator = initialize_calculator(llm)

    
    query = '''SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'employee_details' 
      AND table_schema = 'public'
    ORDER BY ordinal_position;
    '''
    dataset_columns = python_tool.run(query)

    tools = [
        Tool(
            name="Timekeeping Policies",
            func=timekeeping_policy.run,
            description="""
            Useful for when you need to answer questions about employee timekeeping policies and to get the leave encash criteria.

            Below is the example conversation.
            <user>: What is the policy on unused vacation leave?
            <assistant>
                Thought: I need to check the timekeeping policies to answer this question.
                Action: Timekeeping Policies
                Action Input: Vacation Leave Policy - Unused Leave
                ...
            </assistant>
            """
        ),
        Tool(
            name="Employee Data",
            func=python_tool.run,
            description = f"""
            Useful for when you need to answer questions about employee data stored in postgres sql dataset. 
            Run postgres sql query operations on dataset employee_details to help you get the right answer.
            'employee_details' has the following columns: {dataset_columns}
            
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
        ),
        Tool(
            name="Calculator",
            func=calculator.run,
            description = f"""
            Useful when you need to do complex math operations or arithmetic.

            Below is the example conversation. Don't use exact values in real time.
            <user>: If I encash my vacation leaves, how much money I will get?
            <assistant>
                Thought: I need to find out Siva Sai's basic pay and the number of unused vacation leaves..
                Action: Employee Data
                Action Input: SELECT basic_pay_in_rs, vacation_leaves_remaining FROM employee_details WHERE LOWER(name) = 'siva sai';
                Observation: basic_pay_in_rs and vacation_leaves
                Thought: I now have the basic pay and vacation leaves. I can proceed with time keeping policy to find encashment criteris.
                Action: Timekeeping Policies
                Action Input: Leave Encashment Criteria - Vacation Leave
                Observation: Found the encashment criteria is (basic_pay/30)*vacation_leaves
                Thought: Now I can calculate the final amount
                Final Answer: If you encash your n unused vacation leaves, you will be paid x INR.
            </assistant>
            
            """
        )
    ]
    
    return tools, llm

chat_history = []

def add_to_chat_history(user_input, model_response):
    # Format the conversation as a single string with user input and model response
    conversation = f"User: {user_input}\nAssistant: {model_response}"
    
    # Append the conversation to chat history
    chat_history.append(conversation)
    
    # Maintain only the last 2 conversations
    if len(chat_history) > 2:
        chat_history.pop(0)  # Remove the oldest conversation

# Initialize agent
def initialize_agent_with_tools(api_key, user_name):
    #df = load_employee_data()
    tools, llm = initialize_tools(api_key)

    prefix = f'You are friendly HR assistant. You are tasked to assist the current user: {user_name} on questions related to HR. You have access to the following tools:'

    FORMAT_INSTRUCTIONS = f"""
    Use the following format:

    When you have a response to say to the Human, or if you do not need to use a tool, or Observation: is Invalid Format: Missing 'Action:' after 'Thought:, you MUST use the format:
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]

    When you have to use tool, follow below format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [Timekeeping Policies, Employee Data, Calculator], or skip this step if no action is needed
    Action Input: the input to the action (skip this step and give final answer if no action is needed)
    Observation: the result of the action (skip if no action was taken)
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Chat History:
    {"\n\n".join(chat_history)}
    """

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True, 
        agent_kwargs={
            'prefix':prefix,
            'format_instructions':FORMAT_INSTRUCTIONS
        },
        handle_parsing_errors=True
    )

    #print("Prompt========",agent.agent.llm_chain.prompt.template)
    
    return agent


