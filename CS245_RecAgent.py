import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM, DeepseekLLM, OllamaLLM, GeminiLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryBase, MemoryDILU
from websocietysimulator.agent.modules.tooluse_modules import ToolUseToolFormer
from websocietysimulator.tools.evaluation_tool import RecommendationEvaluator
from langchain_core.documents import Document
import re
import ast
import logging
import time
import os
import logging
from dotenv import load_dotenv



def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except Exception:
        print(encoding.encode(string))
        a = 0
    return a


# =========================
#      PLANNING MODULE
# =========================

class RecPlanning(PlanningBase):
    """
    Planning module for the CS245 recommendation agent.

    Features:
      - Generates multiple candidate plans and uses the LLM as a critic
        to select the best one (multi-planning + selection).
      - Uses STRICT, EXCLUSIVE keyword constraints so downstream code
        can parse steps reliably.
      - Can condition on global feedback to refine planning over runs.
    """

    def __init__(self, llm: LLMBase, num_candidate_plans: int = 2, max_tokens: int = 1500):
        super().__init__(llm=llm)
        self.num_candidate_plans = max(1, num_candidate_plans)
        self.max_tokens = max_tokens

    def __call__(self, task_type: str, task_description: str,
                 feedback: str = '', few_shot: str = '') -> list[dict]:
        """
        Main entry point. Generates multiple candidate plans, then selects
        the best one using an LLM-based critic.
        Returns a single chosen plan as a list of steps (dicts with 'description').
        """
        candidate_plans = []
        print("Generating candidate plans...")
        for i in range(self.num_candidate_plans):
            prompt = self._build_planning_prompt(
                task_type=task_type,
                task_description=task_description,
                feedback=feedback,
                few_shot=few_shot,
                candidate_index=i,
            )

            llm_output = self.llm(
                messages=[
                    {
                        "role": "system",
                        "content": ("""
                            You are the PLANNING MODULE for a recommendation agent.
                            Your job is ONLY to create a JSON plan for how the agent should proceed.
                            """ 
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,  # small diversity to get different plans
                max_tokens=self.max_tokens,
            )

            plan_dict = self._parse_plan_from_llm_output(llm_output)
            if plan_dict["steps"]:
                candidate_plans.append(plan_dict)

        # If no candidate plan parses correctly, fall back to a safe static plan.
        if not candidate_plans:
            return self._default_plan()

        # If only one candidate, just use it.
        if len(candidate_plans) == 1:
            return candidate_plans[0]["steps"]

        # Otherwise, use LLM as critic to select best plan.
        best_idx = self._select_best_plan_index(
            candidate_plans=candidate_plans,
            task_type=task_type,
            task_description=task_description,
            feedback=feedback,
        )
        best_idx = max(0, min(best_idx, len(candidate_plans) - 1))
        print("Selected plan index:", best_idx)
        return candidate_plans[best_idx]["steps"]

    # ---- Prompt construction ----

    def _build_planning_prompt(
        self,
        task_type: str,
        task_description: str,
        feedback: str,
        few_shot: str,
        candidate_index: int = 0,
    ) -> str:
        """
        Build the planning prompt with STRICT, EXCLUSIVE keyword constraints.
        """
        print("Building planning prompt for candidate index:", candidate_index)

        base_instructions = """
        You are a planner who divides a {task_type} into several clear sub-tasks.
        Each sub-task should describe a concrete action the agent should take.

        You are generating CANDIDATE PLAN #{candidate_index}.

        DOWNSTREAM MODULES WILL PARSE THE STEP DESCRIPTIONS USING SIMPLE KEYWORD RULES.
        So you MUST follow these STRICT and EXCLUSIVE constraints:

          1. Steps about USER information:
            - The description MUST contain the word "user".
            - The description MUST NOT contain the words "item" or "review".
            - Example (valid): "First I need to gather user information"
            - Example (invalid): "I need to get user and item data"

          2. Steps about ITEM information:
            - The description MUST contain the word "item".
            - The description MUST NOT contain the words "user" or "review".
            - Example (valid): "Next, I need to gather item information"
            - Example (invalid): "I need to inspect user-item interactions"

          3. Steps about REVIEW information:
            - The description MUST contain the word "review".
            - The description MUST CLEARLY INDICATE what type of review information is being gathered (user reviews or item reviews). DO NOT include BOTH user and item reviews in the same step.
            - Example (valid): "Next, I need to gather review information on the user"
            - Example (invalid): "I need to check user and item review history"

          4. Steps about designing or applying a ranking method:
            - The description MUST NOT contain the words "user", "item", or "review".
            - Example (valid): "Next, I need to design a ranking method based on the collected information"
            - Example (valid): "Finally, I need to apply the ranking method to produce the final ranked list"

          5. The plan should have between 4 and 8 sub-tasks, and MUST include:
            - At least one USER-only step (rule 1)
            - At least one ITEM-only step (rule 2)
            - At least one REVIEW-only step (rule 3)
            - At least one ranking-related step (rule 4)

        The plan is for a recommendation scenario where the agent will:
          1) gather user information,
          2) gather candidate item information,
          3) gather review information,
          4) design a ranking method,
          5) apply the ranking method.

        Your output MUST be valid JSON with the following structure:

        {{
          "rationale": "short natural language rationale",
          "steps": [
            {{
              "description": "First I need to gather user information",
              "reasoning_instruction": "optional reasoning guidance for this step"
            }},
            {{
              "description": "Next, I need to gather item information",
              "reasoning_instruction": "..."
            }},
            {{
              "description": "Next, I need to gather review information",
              "reasoning_instruction": "..."
            }},
            {{
              "description": "Next, I need to design a ranking method based on the collected information",
              "reasoning_instruction": "..."
            }},
            {{
              "description": "Finally, I need to apply the ranking method to produce the final ranked list",
              "reasoning_instruction": "..."
            }}
          ]
        }}

        Only output JSON. Do NOT include any extra text outside the JSON.
        """.format(task_type=task_type, candidate_index=candidate_index + 1)

        print("Base instructions prepared.")
        if feedback:
            prompt = f"""{base_instructions}

            Here is feedback from previous attempts at similar tasks (Reflexion):
            \"\"\"{feedback}\"\"\" 

            Use this feedback to improve this candidate plan.

            Current task description:
            \"\"\"{task_description}\"\"\" 
            """
        else:
            prompt = f"""{base_instructions}

            Current task description:
            \"\"\"{task_description}\"\"\" 
            """
        print("Prompt with feedback prepared.")
        if few_shot:
            prompt += f"""

            You may find the following example(s) helpful:
            {few_shot}
            """

        return prompt

    # ---- Parsing helpers ----

    def _parse_plan_from_llm_output(self, llm_output: str) -> dict:
        """
        Parse the LLM output into a dict with keys:
          - "rationale": str
          - "steps": list[{"description": ..., "reasoning_instruction": ...}]
        """
        text = llm_output.strip()
        data = None
        print("Parsing plan from LLM output:", text)
        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                snippet = text[start: end + 1]
                try:
                    data = json.loads(snippet)
                except Exception:
                    data = None

        if not isinstance(data, dict):
            return {"rationale": "Parsing failed", "steps": []}

        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            return {"rationale": "No steps list", "steps": []}

        rationale = str(data.get("rationale", "")).strip()

        steps: list[dict] = []
        for step in raw_steps:
            desc = str(step.get("description", "")).strip()
            if not desc:
                continue
            steps.append(
                {
                    "description": desc,
                    "reasoning_instruction": step.get("reasoning_instruction", ""),
                }
            )

        return {"rationale": rationale, "steps": steps}

    def _select_best_plan_index(
        self,
        candidate_plans: list[dict],
        task_type: str,
        task_description: str,
        feedback: str,
    ) -> int:
        """
        Ask the LLM to pick the best plan among candidate_plans.
        Returns an integer index into candidate_plans.
        """
        print("Selecting best plan among", len(candidate_plans), "candidates...")
        # Compact representation for the critic
        critic_input = []
        for i, plan in enumerate(candidate_plans):
            critic_input.append(
                {
                    "id": i,
                    "rationale": plan["rationale"],
                    "steps": [s["description"] for s in plan["steps"]],
                }
            )

        critic_input_str = json.dumps(critic_input, ensure_ascii=False, indent=2)

        prompt = f"""
        You are evaluating planning strategies for a recommendation agent.

        Task type:
        {task_type}

        Task description:
        \"\"\"{task_description}\"\"\" 

        Previous feedback (may be empty):
        \"\"\"{feedback}\"\"\" 

        Candidate plans (JSON list):
        {critic_input_str}

        For each plan, consider:
          - Does it clearly separate user / item / review steps?
          - Does it follow the required keyword constraints?
          - Does it gather enough information before designing a ranking method?
          - Does it design and apply a reasonable ranking step?

        Return ONLY a JSON object:

        {{
          "best_id": <int>,           // index of the best plan in the list
          "justification": "<why this plan is best>"
        }}
        """
        critic_output = self.llm(
            messages=[
                {"role": "system", "content": "You are a critical evaluator of planning strategies."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )

        text = critic_output.strip()
        best_id = 0

        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                snippet = text[start: end + 1]
                try:
                    data = json.loads(snippet)
                except Exception:
                    data = {}
            else:
                data = {}

        if isinstance(data, dict) and "best_id" in data:
            try:
                best_id = int(data["best_id"])
            except Exception:
                best_id = 0

        return best_id

    def _default_plan(self) -> list[dict]:
        """
        Fallback static plan obeying exclusive keyword rules:

          - Step 1: user-only ("user" only)
          - Step 2: item-only ("item" only)
          - Step 3: review-only ("review" only)
          - Step 4: ranking (no user/item/review)
          - Step 5: ranking (no user/item/review)
        """
        return [
            {"description": "First I need to gather user information"},
            {"description": "Next, I need to gather item information"},
            {"description": "Next, I need to gather review information"},
            {
                "description": "Next, I need to design a ranking method based on the collected information"
            },
            {
                "description": "Finally, I need to apply the ranking method to produce the final ranked list"
            },
        ]

# =========================
#       MEMORY MODULE
# =========================

class RecMemory(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='recall', llm=llm)

    def retriveMemory(self, query_scenario: str):
        task_name = query_scenario

        if self.scenario_memory._collection.count() == 0:
            return ''

        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1
        )

        task_trajectories = [
            f'{result[0].page_content} \n {str(result[0].metadata)}' for result in similarity_results
        ]

        return '\n'.join(task_trajectories)
    
    def addMemory(self, page_content: str, metadata: dict):
        # Create document with metadata
        memory_doc = Document(
            page_content=page_content,
            metadata=metadata
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

# =========================
#      REASONING MODULE
# =========================

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm, tools, max_tokens: int = 12288, memory: RecMemory = None):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)
        self.tools = tools
        self.max_tokens = max_tokens

    def __call__(self, user, items, task_description: str, plan: list[dict]):
        """Override the parent class's __call__ method"""
        reasoning_process = {}
        reasoning_result = "[]"

        memory_query = {"user": user, "items": items}
        memory = self.memory.retriveMemory(json.dumps(memory_query)) if self.memory else ''
        memory_prompt = f"Here is relevant memory from past task(s) to help improve your reasoning:\n{memory}\n" if memory else ''
        print("Memory:", memory_prompt)
        for step in plan:
            print("Sub-task:", step['description'])
            thinkingStrings = ['design', 'apply', 'method', 'rank']
            thinkingStep = any(thinkingString in step.get('description', '') for thinkingString in thinkingStrings)
            llm_output = self.llm(
                messages=[
                    {"role": "assistant", "content": str(reasoning_process) + '\n' + memory_prompt},
                    {"role": "system", "content": task_description},
                    {"role": "user", "content": step.get('description', '') + '\n' + step.get('reasoning_instruction', '')},
                ],
                temperature=0.1,
                # max_tokens=self.max_tokens,
                max_tokens=self.max_tokens*2 if thinkingStep else self.max_tokens,
                response_format={"type": "json_object"},
            )
            print("LLM Output:", llm_output)
            try:
                action = json.loads(llm_output)
            except:
                action = llm_output
            reasoning_process[step['description']] = [action]
            if isinstance(action, dict) and 'action' in action and action['action'] in self.tools:
                tool_name = action['action']
                tool_input = action['action_input']
                tool_output = self.tools[tool_name]['function'](**tool_input)
                print(f"Tool used: {tool_name}, Input: {tool_input}, Output: {tool_output}")
                reasoning_process[step['description']].append(tool_output)
            elif isinstance(action, dict) and 'action' in action and action['action'] == 'FINISH':
                reasoning_result = str(action.get('ranked_ids', '[]'))
                print("Final answer reached.", reasoning_result)
                break
        summary = self.summarize_trajectory(json.dumps(reasoning_process))
        return {"result": reasoning_result, "summary": summary}
    
    def summarize_trajectory(self, trajectory: str):
        summary_prompt = f"""
        You are summarizing an agent's recommendation task trajectory. Write a concise summary highlighting key actions taken and decisions made.
        Your summary should include:
          - Major steps the agent took (e.g., gathering user/item/review info, designing ranking method)
          - What user preferences were inferred
          - What ranking strategy was used
          The summary should be at most 4 consice sentences.
          Here is the trajectory:
          {trajectory}
          Now output the summary:
        """
        summary = self.llm(messages=[{"role": "user", "content": summary_prompt}], temperature=0.1, max_tokens=512)
        return summary.strip()

# =========================
#      RECOMMENDATION AGENT
# =========================

class RecommendationAgentCS245(RecommendationAgent):
    # Global feedback shared across agent instances (global refinement)
    GLOBAL_FEEDBACK: str = ""

    def __init__(self, llm: LLMBase, memory: RecMemory = None):
        super().__init__(llm=llm)
        # each agent instance reads the current global feedback
        self.global_feedback = RecommendationAgentCS245.GLOBAL_FEEDBACK
        self.planning = RecPlanning(llm=self.llm)
        self.tools = {}
        self.reasoning: RecReasoning | None = None
        self.memory = memory

    def set_interaction_tool(self, interaction_tool):
        super().set_interaction_tool(interaction_tool)
        self.tools = {
            "get_user": {
                "function": self.interaction_tool.get_user,
                "description": "Fetch user data based on user_id",
                "parameters": {"user_id": "str"},
            },
            "get_item": {
                "function": self.interaction_tool.get_item,
                "description": "Fetch item data based on item_id",
                "parameters": {"item_id": "str"},
            },
            "get_items": {
                "function": self.interaction_tool.get_items,
                "description": "Fetch multiple items based on a list of item_ids",
                "parameters": {"item_ids": "List[str]"},
            },
            "get_reviews": {
                "function": self.interaction_tool.get_reviews,
                "description": "Fetch reviews filtered by various parameters",
                "parameters": {
                    "item_ids": "Optional[List[str]]",
                    "item_id": "Optional[str]",
                    "user_id": "Optional[str]",
                    "review_id": "Optional[str]",
                },
            },
        }
        self.reasoning = RecReasoning(
            profile_type_prompt='', llm=self.llm, tools=self.tools, memory=self.memory
        )

    def workflow(self) -> list[dict[str, any]]:
        user_id = self.task['user_id']
        candidate_list = self.task['candidate_list']

        simulation_config = {
            "num_candidate_plans": 1,
            "max_reasoning_tokens": 4096,
            "max_planning_tokens": 1024,
        }

        self.planning.num_candidate_plans = simulation_config["num_candidate_plans"]
        self.reasoning.max_tokens = simulation_config["max_reasoning_tokens"]
        self.planning.max_tokens = simulation_config["max_planning_tokens"]
        # --- PLANNING: multi-plan + selection, with global feedback ---

        plan_task_description = f"""
        Please make a plan to rank a list of candidate items for a given user.

        You will later have access to tools:
        - get_user(user_id) to fetch user information
        - get_item(item_id) / get_items(item_ids) to fetch item information
        - get_reviews(user_id=..., item_id=...) to fetch review information

        You are given a user with id: {user_id} and a list of candidate item IDs: {candidate_list}.
        Your plan should describe:
          1) how to obtain user information,
          2) how to obtain candidate item information,
          3) how to obtain review information (for user and/or items),
          4) how to design a ranking method based on this information,
          5) how to apply the ranking method to produce a ranked list.
        """
        plan = self.planning(
            task_type='Recommendation Task',
            task_description=plan_task_description,
            feedback=self.global_feedback,
            few_shot='',
        )
        print("Generated plan:", plan)

        # --- INFORMATION GATHERING BASED ON PLAN ---

        user = ''
        item_list = []
        history_review = ''

        for sub_task in plan:
            desc = sub_task.get('description', '')

            if 'user' in desc.lower():
                user = str(self.interaction_tool.get_user(user_id=user_id))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])

            elif 'item' in desc.lower():
                item_list = []
                for item_id in candidate_list:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    keys_to_extract = [
                        'item_id',
                        'name',
                        'stars',
                        'review_count',
                        'attributes',
                        'title',
                        'average_rating',
                        'rating_number',
                        'description',
                        'ratings_count',
                        'title_without_series',
                    ]
                    filtered_item = {
                        key: item[key] for key in keys_to_extract if key in item
                    }
                    item_list.append(filtered_item)

            elif 'review' in desc.lower():
                history_review = str(
                    self.interaction_tool.get_reviews(user_id=user_id)
                )
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(
                        encoding.encode(history_review)[:12000]
                    )
            else:
                # ranking / meta steps, no environment calls here
                pass

        # --- TASK DESCRIPTION FOR FINAL REASONING / RANKING ---

        reasoning_task_description = f"""
        You are a recommendation system tasked with ranking a list of candidate items for a user based on their preferences. You are given the user: {user_id} and a list of candidate items to rank: {candidate_list}.
        
        You can use the tools {self.tools} to gather necessary information about the user and items. If you use a tool, you MUST specify the tool name under "action" and input parameters under "action_input". 
        The tool name MUST match exactly with one of the tool names provided. Make sure the input parameters are in the correct format as expected by the tool. 
        The information about each tool is included in the tool descriptions and parameter information is included as well.

        You are also given a plan you should follow. For each sub-task in the plan, you should create an action and execute it. For example, if the sub-task is to gather user information, you should create an action that uses the get_user tool with the appropriate user_id.
        You should also include your thoughts and reasoning for each action you take. Your reasoning should be relatively concise. 
        When action is FINISH, ranked_ids MUST ALSO be populated. You must include all the candidate ids in your final ranked_ids list.
        
        OUTPUT FORMAT:
        {{
          "thoughts": "Your reasoning here",
          "action": "tool_name or FINISH", 
          "action_input": {{input1: "value1", input2: "value2", ...}},
          "ranked_ids": ["id1", "id2", ...]  // only if action is FINISH
        }}    
        NO code blocks, NO backticks, NO commentary. YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT. DO NOT INCLUDE ANY CONVERSATIONAL TEXT, EXPLANATIONS, OR MARKDOWN OUTSIDE THE JSON. ONLY OUTPUT THE JSON.
        If you deviate from the format even slightly, I will terminate the run. Output ONLY the format.”
        """

        result = self.reasoning(
            user=user_id,
            items=candidate_list,
            plan=plan,
            task_description=reasoning_task_description,
        )

        # Parse result, get summary for memory
        summary = result.get('summary', '')
        result = str(result.get('result', '[]'))

        print("candidate list: ", candidate_list)
        print("result: ", result, type(result))
        print("item_list: ", item_list)
        print("history_review: ", history_review)
        print("user: ", user)
        # --- POST-PROCESSING OF LLM OUTPUT ---
        parsed_result = None
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                print("List found in result.", match)
                result_list_str = match.group()
            else:
                print("No list found.")
                return ['']
            try:
                processed = ast.literal_eval(result_list_str)
                print('Processed Output:', processed)
                parsed_result = processed
            except Exception:
                print('literal_eval failed, falling back to eval')
                parsed_result = eval(result_list_str)
        except Exception as e:
            print('format error', e)
            parsed_result = ['']
        
        # Collect metadata and summary for memory
        metadata = {
            "user_id": str(user_id),
            "candidate_list": str(candidate_list),
            "final_ranking": str(parsed_result),
        }

        return {"result": parsed_result, "metadata": metadata, "page_content": summary}


# =========================
#          MAIN
# =========================

if __name__ == "__main__":
    task_set = "yelp"  # "goodreads" or "yelp"
    num_tasks = 25     # adjust if you want more

    load_dotenv()
    HF_TOKEN = os.environ.get("HF_TOKEN")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # -------- PHASE 1: initial run to gather feedback --------
    # setup
    simulator1 = Simulator(data_dir="processed_datasets", device="auto", cache=True)
    simulator1.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks",
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth",
    )

    tasks = simulator1.tasks[:num_tasks]
    groundtruths = [gt['ground truth'] for gt in simulator1.groundtruth_data]
    # llm = GeminiLLM(api_key=GEMINI_API_KEY)
    llm = OllamaLLM(model_name="deepseek-r1")  # Example for local Ollama server
    memory = RecMemory(llm=llm)
    agent = RecommendationAgentCS245(llm=llm, memory=memory)
    evaluator = RecommendationEvaluator()
    logger = logging.getLogger("websocietysimulator")

    predictions = []
    # run tasks one by one to gather memory
    for i in range(len(tasks)):
        task = tasks[i]
        groundtruth = groundtruths[i]
        agent.set_interaction_tool(simulator1.interaction_tool)
        agent.insert_task(task)
        output = {}
        try:
            output = agent.workflow()
            # set up memory
            evaluation = evaluator.calculate_hr_at_n(
                ground_truth=[groundtruth],
                predictions=[output.get('result', [])],
            )
            metadata = output.get('metadata', {})
            metadata['evaluation'] = str(evaluation)
            print("Agent output:", output)
            print(f"Evaluation for task {i}:", evaluation)
            memory.addMemory(page_content=output.get('summary', ''), metadata=metadata)
            logger.info(f"Simulation finished for task {i}")
        except Exception as e:
            logger.error(f"Task {i} failed with error: {str(e)}")
        
        # add output for final evaluation
        predictions.append(output.get('result', []))
    
    logger.info("Simulation finished")


    # simulator1.set_agent(RecommendationAgentCS245)
    # simulator1.set_llm(GeminiLLM(api_key=GEMINI_API_KEY))  # or another LLMBase subclass

    # agent_outputs_1 = simulator1.run_simulation(
    #     number_of_tasks=num_tasks, enable_threading=True, max_workers=10
    # )

    # evaluation_results_1 = simulator1.evaluate()
    evaluation_results_1 = evaluator.calculate_hr_at_n(
        ground_truth=groundtruths[:num_tasks],
        predictions=predictions,
    )
    evaluation_results_1 = {'type': 'recommendation', 'results': evaluation_results_1.__dict__}
    evaluation_results_1['data_info'] = {
            'evaluated_count': min(num_tasks, len(groundtruths)),
            'original_simulation_count': num_tasks,
            'original_ground_truth_count': len(groundtruths)
    }
        
    with open(f'./evaluation_results_track2_{task_set}_phase1.json', 'w') as f:
        json.dump(evaluation_results_1, f, indent=4)

    print(f"[PHASE 1] evaluation_results: {evaluation_results_1}")

    # Build a simple global feedback string for the planner
    feedback_str = (
        "Global evaluation feedback from previous run. "
        "Raw metrics JSON: " + json.dumps(evaluation_results_1)
        + ". Plans should try to improve hit rates at top ranks "
          "and better align recommendations with observed user behavior."
    )

    # Set global feedback so future agent instances can read it
    RecommendationAgentCS245.GLOBAL_FEEDBACK = feedback_str

    # -------- PHASE 2: refined planning and final evaluation --------

    # simulator2 = Simulator(data_dir="processed_datasets", device="auto", cache=True)
    # simulator2.set_task_and_groundtruth(
    #     task_dir=f"./example/track2/{task_set}/tasks",
    #     groundtruth_dir=f"./example/track2/{task_set}/groundtruth",
    # )

    # simulator2.set_agent(RecommendationAgentCS245)
    # simulator2.set_llm(GeminiLLM(GEMINI_API_KEY))  # or another LLMBase subclass

    # agent_outputs_2 = simulator2.run_simulation(
    #     number_of_tasks=num_tasks, enable_threading=True, max_workers=10
    # )

    # evaluation_results_2 = simulator2.evaluate()
    # with open(f'./evaluation_results_track2_{task_set}_phase2.json', 'w') as f:
    #     json.dump(evaluation_results_2, f, indent=4)

    # print(f"[PHASE 2] evaluation_results: {evaluation_results_2}")
