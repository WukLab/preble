from transformers import AutoTokenizer
from benchmarks.benchmark_workload_gen import get_react_workload

# model_name = "lmsys/vicuna-13b-v1.5"
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_workloads = 10
prompt = """
System: You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:
1.whatsapp_api: To Send Messages From WhatsApp

Specifically, you have access to the following APIs: [{'name': 'phonelist_for_whatsapp_api', 'description': 'This is the subfunction for tool "whatsapp_api", you can use this tool.', 'parameters': {'type': 'object', 'properties': {'product_id': {'type': 'string', 'description': '', 'example_value': 'product_id'}}, 'required': ['product_id'], 'optional': []}}, {'name': 'logs_for_whatsapp_api', 'description': 'This is the subfunction for tool "whatsapp_api", you can use this tool.', 'parameters': {'type': 'object', 'properties': {'product_id': {'type': 'string', 'description': '', 'example_value': 'product_id'}}, 'required': ['product_id'], 'optional': []}}, {'name': 'productdata_for_whatsapp_api', 'description': 'This is the subfunction for tool "whatsapp_api", you can use this tool.', 'parameters': {'type': 'object', 'properties': {'product_id': {'type': 'string', 'description': '', 'example_value': 'product_id'}}, 'required': ['product_id'], 'optional': []}}, {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"'}}, 'required': ['return_type']}}]
User: 
"""
print(len(tokenizer.encode(prompt)))
