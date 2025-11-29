import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM, DeepseekLLM, OllamaLLM, GeminiLLM
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

    def __init__(self, llm: LLMBase, num_candidate_plans: int = 2):
        super().__init__(llm=llm)
        self.num_candidate_plans = max(1, num_candidate_plans)

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
                max_tokens=2000,
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
            - The description MUST NOT contain the words "user" or "item".
            - Example (valid): "Next, I need to gather review information"
            - Example (invalid): "I need to check user review history"

          4. Steps about designing or applying a ranking method:
            - The description MUST NOT contain the words "user", "item", or "review".
            - Example (valid): "Next, I need to design a ranking method based on the collected information"
            - Example (valid): "Finally, I need to apply the ranking method to produce the final ranked list"

          5. The plan should have between 4 and 6 sub-tasks, and MUST include:
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
            max_tokens=1000,
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
#      REASONING MODULE
# =========================

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm, tools):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        self.tools = tools

    def __call__(self, user, items, task_description: str, plan: list[dict]):
        """Override the parent class's __call__ method"""
        reasoning_process = {}
        for step in plan:
            print("Sub-task:", step['description'])
            llm_output = self.llm(
                messages=[
                    {"role": "assistant", "content": str(reasoning_process)},
                    {"role": "system", "content": task_description},
                    {"role": "user", "content": step.get('description', '') + '\n' + step.get('reasoning_instruction', '')},
                ],
                temperature=0.1,
                max_tokens=24576,
                response_format={"type": "json_object"},
            )
            print("LLM Output:", llm_output)
            action = json.loads(llm_output)
            reasoning_process[step['description']] = [action]
            if 'tool' in action and action['tool'] in self.tools:
                tool_name = action['tool']
                tool_input = action['tool_input']
                tool_output = self.tools[tool_name]['function'](**tool_input)
                print(f"Tool used: {tool_name}, Input: {tool_input}, Output: {tool_output}")
                reasoning_process[step['description']].append(tool_output)
        
        final_system = {
            "role": "system",
            "content": """
                You are a recommendation agent. OUTPUT EXACTLY ONE JSON OBJECT and ONLY JSON.
                Do NOT output any text, code, or markdown. Do NOT output Python or code fences. 
                The JSON MUST match this schema exactly: 
                {{ 'analysis': '<string>', 'scores': [{{'id': '<string>', 'score': <int>, 'justification': '<short>'}}, ...], 'ranked_ids': ['id1','id2',...] }}
            """
        }

        final_assistant = {"role": "assistant", "content": str(reasoning_process)}

        final_user = {
            "role": "user",
            "content": f"""
                Use your previous reasoning process to rank the candidate items for the user.
                Candidate item IDs: {json.dumps(items)}
                For each candidate, assign an integer score between 0 and 100 (higher is better) based on how likely the user would highly rate the candidate, give a one-line justification,
                and produce ranked_ids ordered by score descending. Output ONLY the final JSON object:
                {{ 'analysis': '<string>', 'scores': [{{'id': '<string>', 'score': <int>, 'justification': '<short>'}}, ...], 'ranked_ids': ['id1','id2',...] }}
                NOTHING else. For instance:
                {{
                  'analysis': 'Based on the user preferences and item features, I evaluated each candidate as follows...',
                  'scores': [
                    {{'id': 'item_123', 'score': 95, 'justification': 'Highly matches user preferences for fine dining'}},
                    {{'id': 'item_456', 'score': 80, 'justification': 'Good match but lacks some features such as outdoor seating'}},
                    ...
                  ],
                  'ranked_ids': ['item_123', 'item_456', ...]
                }}
                YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT. DO NOT INCLUDE ANY CONVERSATIONAL TEXT, EXPLANATIONS, OR MARKDOWN OUTSIDE THE JSON. ONLY OUTPUT THE JSON.
                If you deviate from the format even slightly, I will terminate the run. Output ONLY the format.”
            """
        }
        reasoning_result = self.llm(messages=[final_assistant, final_system, final_user], temperature=0.1, max_tokens=24576)
        
        return reasoning_result


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
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]

        return '\n'.join(task_trajectories)


# =========================
#      RECOMMENDATION AGENT
# =========================

class RecommendationAgentCS245(RecommendationAgent):
    # Global feedback shared across agent instances (global refinement)
    GLOBAL_FEEDBACK: str = ""

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        # each agent instance reads the current global feedback
        self.global_feedback = RecommendationAgentCS245.GLOBAL_FEEDBACK
        self.planning = RecPlanning(llm=self.llm, num_candidate_plans=2)
        self.tools = {}
        self.reasoning: RecReasoning | None = None

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
                    "item_id": "Optional[str]",
                    "user_id": "Optional[str]",
                    "review_id": "Optional[str]",
                },
            },
        }
        self.reasoning = RecReasoning(
            profile_type_prompt='', llm=self.llm, tools=self.tools
        )

    def workflow(self) -> list[dict[str, any]]:
        user_id = self.task['user_id']
        candidate_list = self.task['candidate_list']

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
        
        You can use the tools {self.tools} to gather necessary information about the user and items. If you use a tool, you MUST specify the tool name under "tool" and input parameters under "tool_input". 
        The tool name MUST match exactly with one of the tool names provided. Make sure the input parameters are in the correct format as expected by the tool. 
        The information about each tool is included in the tool descriptions and parameter information is included as well.

        You are also given a plan you should follow. For each sub-task in the plan, you should create an action and execute it. For example, if the sub-task is to gather user information, you should create an action that uses the get_user tool with the appropriate user_id.
        You should also include your thoughts and reasoning for each action you take.
        
        OUTPUT FORMAT:
        {{
          "thoughts": "Your reasoning here",
          "tool": "tool_name", 
          "tool_input": {{input1: "value1", input2: "value2", ...}},
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

        print("candidate list: ", candidate_list)
        print("result: ", result)
        print("item_list: ", item_list)
        print("history_review: ", history_review)
        print("user: ", user)

        # --- POST-PROCESSING OF LLM OUTPUT ---

        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result_list_str = match.group()
            else:
                print("No list found.")
                return ['']
            print('Processed Output:', eval(result_list_str))
            return eval(result_list_str)
        except Exception:
            print('format error')
            return ['']


# =========================
#          MAIN
# =========================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task_set = "yelp"  # "goodreads" or "yelp"
    num_tasks = 1      # adjust if you want more

    HF_TOKEN = os.environ.get("HF_TOKEN")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # -------- PHASE 1: initial run to gather feedback --------

    simulator1 = Simulator(data_dir="processed_datasets", device="auto", cache=True)
    simulator1.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks",
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth",
    )

    simulator1.set_agent(RecommendationAgentCS245)
    simulator1.set_llm(GeminiLLM(GEMINI_API_KEY))  # or another LLMBase subclass

    agent_outputs_1 = simulator1.run_simulation(
        number_of_tasks=num_tasks, enable_threading=True, max_workers=10
    )

    evaluation_results_1 = simulator1.evaluate()
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

    simulator2 = Simulator(data_dir="processed_datasets", device="auto", cache=True)
    simulator2.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks",
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth",
    )

    simulator2.set_agent(RecommendationAgentCS245)
    simulator2.set_llm(GeminiLLM(GEMINI_API_KEY))  # or another LLMBase subclass

    agent_outputs_2 = simulator2.run_simulation(
        number_of_tasks=num_tasks, enable_threading=True, max_workers=10
    )

    evaluation_results_2 = simulator2.evaluate()
    with open(f'./evaluation_results_track2_{task_set}_phase2.json', 'w') as f:
        json.dump(evaluation_results_2, f, indent=4)

    print(f"[PHASE 2] evaluation_results: {evaluation_results_2}")
