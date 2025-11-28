import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM, DeepseekLLM, OllamaLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryBase
from websocietysimulator.agent.modules.tooluse_modules import ToolUseToolFormer
import re
import logging
import time
import torch
import os

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
            The following are some examples:
            Task: I need to find some information to complete a recommendation task.
            sub-task 1: {{"step": 1, "description": "First I need to find user information", "reasoning instruction": "None"}}
            sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
            sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

            Task: {task_description}
            '''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
            The following are some examples:
            Task: I need to find some information to complete a recommendation task.
            sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
            sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
            sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

            end
            --------------------
            Reflexion:{feedback}
            Task:{task_description}
            '''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm, tools):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        self.tools = tools
        
    def __call__(self, user, items, task_description: str, plan: str):
        """Override the parent class's __call__ method"""
        prompt = '''
        {task_description}
        '''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )

        reasoning_process = {}
        for step in plan:
            print("Sub-task:", step['description'])
            llm_output = self.llm(
                messages=[
                    {"role": "system", "content": task_description},
                    {"role": "user", "content": step['description']}],
                temperature=0.1,
                max_tokens=1000,
                response_format={'type': 'json_object'})
            print("LLM Output:", llm_output)
            action = json.loads(llm_output)
            reasoning_process[step['description']] = [action]
            if 'tool' in action and action['tool'] in self.tools:
                tool_name = action['tool']
                tool_input = action['tool_input']
                tool_output = self.tools[tool_name]['function'](**tool_input)
                print(f"Tool used: {tool_name}, Input: {tool_input}, Output: {tool_output}")
                reasoning_process[step['description']].append(tool_output)
            
        reasoning_result = self.llm(
            messages=[
                {"role": "system", "content": "You are a recommendation agent that makes final recommendations based on the reasoning process, and must give output in the required format: [item_id1, item_id2, ...]. Do NOT include any additional text."},
                {"role": "user", "content": f"Please use the reasoning given here: {reasoning_process} to calculate a ranking of item IDs from the candidate items: {items} for the user: {user} in the correct format: [item_id1, item_id2, ...]."}],
                temperature=0.1,
                max_tokens=1000)
        
        return reasoning_result

class RecMemory(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='recall', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query scenario
        task_name = query_scenario
        
        # Return empty string if memory is empty
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
            
        # Extract task trajectories from results
        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]
        
        # Join trajectories with newlines and return
        return '\n'.join(task_trajectories)

    # def addMemory(self, current_situation: str):
    #     # Extract task description
    #     task_name = current_situation
        
    #     # Create document with metadata
    #     memory_doc = Document(
    #         page_content=task_name,
    #         metadata={
    #             "task_name": task_name,
    #             "task_trajectory": current_situation
    #         }
    #     )
        
    #     # Add to memory store
    #     self.scenario_memory.add_documents([memory_doc])

class RecommendationAgentCS245(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.tools = {}
        self.reasoning = None

    def set_interaction_tool(self, interaction_tool):
        super().set_interaction_tool(interaction_tool) # Call base class method
        self.tools = {
            "get_user": {"function": self.interaction_tool.get_user, "description": "Fetch user data based on user_id", "parameters": {"user_id": "str"}},
            "get_item": {"function": self.interaction_tool.get_item, "description": "Fetch item data based on item_id", "parameters": {"item_id": "str"}},
            "get_items": {"function": self.interaction_tool.get_items, "description": "Fetch multiple items based on a list of item_ids", "parameters": {"item_ids": "List[str]"}},
            "get_reviews": {"function": self.interaction_tool.get_reviews, "description": "Fetch reviews filtered by various parameters", "parameters": {"item_id": "Optional[str]", "user_id": "Optional[str]", "review_id": "Optional[str]"}},
        }
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm, tools=self.tools)

    def workflow(self) -> list[dict[str, any]]:
        user_id = self.task['user_id']
        candidate_list = self.task['candidate_list']

        # Retrieve past experience from memory
        # past_experience = self.memory.retriveMemory(current_situation=f"User {user_id} with candidates {candidate_list}")

        # Formulate plan
        plan_task_description = '''
        Please make a plan to rank a list of candidate items for a given user. You are given information on the user, their historical reviews, and on the candidate items.
        '''
        plan = self.planning(task_type='Recommendation Task', task_description=plan_task_description, feedback='', few_shot='')
        print(plan)
        # Reasoning and generate final recommendation


        plan = [
         {'description': 'First I need to find the user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'},
         {'description': 'Finally, I need to come up with a method to rank the items based on the information I have gathered'}
        #  {'description': 'Next, I need to retrieve past experience'}
         ]

        user = ''
        item_list = []
        history_review = ''
        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])
                    keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                item_list.append(filtered_item)
                # print(item)
            elif 'review' in sub_task['description']:
                history_review = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        task_description = f'''
        You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
        Now you need to rank the following 20 items: {self.task['candidate_list']} according to their match degree to your preference.
        Please rank the more interested items more front in your rank list.
        The information of the above 20 candidate items is as follows: {item_list}.

        Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
        DO NOT output your analysis process!

        The correct output format:

        ['item id1', 'item id2', 'item id3', ...]

        '''

        reasoning_task_description = f'''
        You are a recommendation system tasked with ranking a list of candidate items for a user based on their preferences. You are given the user: {user_id} and a list of candidate items to rank: {candidate_list}.
        You can use the tools {self.tools} to gather necessary information about the user and items. If you use a tool, you must specify the tool name and input parameters. The tool name MUST match exactly with one of the tool names provided.
        Make sure the input parameters are in the correct format as expected by the tool. The information about each tool is included in the tool descriptions and parameter information is included as well.
        You are also given a plan you should follow. For each sub-task in the plan, you should create an action and execute it. If you need to use a tool, you NEED to specify the tool name under tool_name and input parameters under tool_input, just specifying the tool under action is NOT enough.
        Otherwise, you should do reasoning on the information you have gathered to produce the final ranked list of item IDs. The format should strictly follow the example below.
        {{
          "thoughts": "Your reasoning here",
          "action": "string", 
          "tool": "tool_name", 
          "tool_input": {{input1: "value1", input2: "value2", ...}},
        }}    
        '''
        result = self.reasoning(user=user_id, items=candidate_list, plan=plan, task_description=reasoning_task_description)
        print("candidate list: ", self.task['candidate_list'])
        print("result: ", result)
        print("item_list: ", item_list)
        print("history_review: ", history_review)
        print("user: ", user)
        try:
            # print('Meta Output:',result)
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
            print('Processed Output:',eval(result))
            # time.sleep(4)
            return eval(result)
        except:
            print('format error')
            return ['']
        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    task_set = "yelp" # "goodreads" or "yelp"
    # Initialize Simulator
    simulator = Simulator(data_dir="processed_datasets", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(task_dir=f"./example/track2/{task_set}/tasks", groundtruth_dir=f"./example/track2/{task_set}/groundtruth")

    # Set your custom agent
    simulator.set_agent(RecommendationAgentCS245)

    # Set LLM client
    HF_TOKEN = os.environ.get("HF_TOKEN")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    simulator.set_llm(OllamaLLM())

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=1, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")