import json
import copy
from typing import Any, Dict, List, Optional, Tuple
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

ALLOWED_STEP_TYPES = {
    "FETCH_USER",
    "FETCH_ITEMS",
    "FETCH_USER_REVIEWS",
    "FETCH_CANDIDATE_REVIEWS",
    "RANK",
}

DEFAULT_FALLBACK_PLAN: Dict[str, Any] = {
    "steps": [
        {"type": "FETCH_USER", "params": {"user_id": "{{user_id}}"}, "reasoning_instruction": ""},
        {"type": "FETCH_ITEMS", "params": {"item_ids": "{{candidate_list}}"}, "reasoning_instruction": ""},
        {"type": "FETCH_USER_REVIEWS", "params": {"user_id": "{{user_id}}"}, "reasoning_instruction": ""},
        {"type": "FETCH_CANDIDATE_REVIEWS", "params": {"item_ids": "{{candidate_list}}"}, "reasoning_instruction": ""},
        {"type": "RANK", "params": {}, "reasoning_instruction": ""},
    ]
}


def _extract_json_objects(text: str) -> List[str]:
    """
    Extract candidate top-level JSON objects from arbitrary text using balanced braces.
    Much safer than greedy regex. Returns a list of JSON object strings.
    """
    objs: List[str] = []
    start = None
    depth = 0
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_str = False
                escape = False
            continue

        # We are inside a candidate object: track strings so braces in strings don't count
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objs.append(text[start : i + 1])
                    start = None

    return objs


def _try_json_loads(s: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Correctly ignores braces inside quoted strings.
    Handles cases where the model prints extra text before/after the JSON.
    Tries multiple extracted objects and prefers one that actually looks like a plan.
    """
    try:
        return json.loads(s), None
    except Exception as e:
        direct_err = str(e)

    candidates = _extract_json_objects(s)

    if not candidates:
        candidates = re.findall(r"\{.*?\}", s, flags=re.DOTALL)

    parse_errors: List[str] = [f"Direct JSON parse error: {direct_err}"]

    for idx, cand in enumerate(candidates):
        try:
            obj = json.loads(cand)
        except Exception as e2:
            parse_errors.append(f"Candidate {idx} parse error: {e2}")
            continue

        # Heuristic: prefer objects that look like a plan
        if isinstance(obj, dict) and "steps" in obj:
            return obj, None
        if isinstance(obj, list):
            return obj, None


    return None, " | ".join(parse_errors)

def _is_user_placeholder(x: Any) -> bool:
    return isinstance(x, str) and x.strip() == "{{user_id}}"

def _is_candidate_placeholder(x: Any) -> bool:
    return isinstance(x, str) and x.strip() == "{{candidate_list}}"

def _validate_step_params(step_type_norm: str, params: Dict[str, Any]) -> Optional[str]:
    if step_type_norm in {"FETCH_USER", "FETCH_USER_REVIEWS"}:
        uid = params.get("user_id")
        if not (isinstance(uid, str) and uid.strip()):
            return "requires params.user_id as a non-empty string (or '{{user_id}}')."
        # allow placeholder explicitly
        if uid.strip() == "{{user_id}}":
            return None
        return None

    if step_type_norm in {"FETCH_ITEMS", "FETCH_CANDIDATE_REVIEWS"}:
        ids = params.get("item_ids")
        if _is_candidate_placeholder(ids):
            return None
        if not isinstance(ids, list) or len(ids) == 0:
            return "requires params.item_ids as a non-empty list of strings (or '{{candidate_list}}')."
        if not all(isinstance(x, str) and x.strip() for x in ids):
            return "requires params.item_ids to be list[str] with non-empty strings."
        return None

    # RANK or unknown handled elsewhere
    return None


def _validate_plan_obj(plan_obj: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Returns a normalized plan dict {"steps":[...]} if valid enough to execute, else (None, errors).
    Accepts:
      - dict with "steps"
      - list of steps
    Normalizes:
      - list -> {"steps": list}
      - step.type uppercased
      - missing params -> {}
      - drops unknown fields without failing
    """
    errors: List[str] = []

    if isinstance(plan_obj, list):
        plan_obj = {"steps": plan_obj}

    if not isinstance(plan_obj, dict):
        return None, ["Plan is not a dict or list."]

    steps = plan_obj.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return None, ["Plan missing non-empty 'steps' list."]

    normalized_steps: List[Dict[str, Any]] = []
    saw_rank = False

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"Step {i} is not an object.")
            continue

        step_type = step.get("type")
        if not isinstance(step_type, str) or not step_type.strip():
            errors.append(f"Step {i} missing string 'type'.")
            continue

        step_type_norm = step_type.strip().upper()
        if step_type_norm not in ALLOWED_STEP_TYPES:
            errors.append(f"Step {i} has invalid type '{step_type}'.")
            continue

        params = step.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            errors.append(f"Step {i} params must be an object.")
            continue

        param_err = _validate_step_params(step_type_norm, params)
        if param_err:
            errors.append(f"Step {i} ({step_type_norm}) {param_err}")
            continue
        # Optional fields
        reasoning_instruction = step.get("reasoning_instruction", "")
        if reasoning_instruction is None:
            reasoning_instruction = ""
        if not isinstance(reasoning_instruction, str):
            errors.append(f"Step {i} reasoning_instruction must be a string.")
            continue

        desc = step.get("description", "")
        if desc is None:
            desc = ""
        if not isinstance(desc, str):
            errors.append(f"Step {i} description must be a string.")
            continue

        if step_type_norm == "RANK":
            saw_rank = True

        normalized_steps.append(
            {
                "type": step_type_norm,
                "params": params,
                "reasoning_instruction": reasoning_instruction,
                "description": desc,
            }
        )

    if not normalized_steps:
        return None, errors or ["No valid steps found."]

    if not saw_rank:
        normalized_steps.append({"type": "RANK", "params": {}, "reasoning_instruction": "", "description": ""})

    return {"steps": normalized_steps}, errors
class RecPlanning(PlanningBase):
    """Planner that outputs typed steps suitable for deterministic execution."""

    def __init__(self, llm):
        super().__init__(llm=llm)

    def create_prompt(self, task_type, task_description, feedback, few_shot):
        schema = r"""
Return ONLY valid JSON using this schema:

{
  "steps": [
    {
      "type": "FETCH_USER",
      "params": {"user_id": "{{user_id}}"},
      "reasoning_instruction": "string",
      "description": "string (optional)"
    },
    {
      "type": "FETCH_ITEMS",
      "params": {"item_ids": ["item1", "item2", ...]},
      "reasoning_instruction": "string",
      "description": "string (optional)"
    }
  ]
}

Allowed type values:
FETCH_USER, FETCH_ITEMS, FETCH_USER_REVIEWS, FETCH_CANDIDATE_REVIEWS, RANK

Rules:
- Do NOT invent user_id or item ids. Use placeholders exactly:
  - "{{user_id}}"
  - "{{candidate_list}}"
- For FETCH_ITEMS and FETCH_CANDIDATE_REVIEWS, put item ids in params.item_ids
- Output JSON only. No markdown, no natural language.
"""

        if not feedback:
            return f"""You are a planner for a {task_type}.
You must produce a typed execution plan for a recommendation agent.

Task description:
{task_description}

{schema}
"""
        return f"""You are a planner for a {task_type}.
You must produce a typed execution plan for a recommendation agent.

Reflection feedback to incorporate:
{feedback}

Task description:
{task_description}

{schema}
"""

    def plan_with_validation(
        self,
        task_type: str,
        task_description: str,
        user_id: str,
        candidate_list: List[str],
        feedback: str = "",
        few_shot: str = "",
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """
        Try up to max_attempts times to get a valid JSON plan.
        If all fail, return DEFAULT_FALLBACK_PLAN.
        """
        prompt = self.create_prompt(task_type, task_description, feedback, few_shot)

        last_errs: List[str] = []
        for attempt in range(1, max_attempts + 1):
            # explicit reminder message for retries.
            retry_suffix = ""
            if attempt > 1:
                retry_suffix = (
                    "\n\nYour previous output was invalid. "
                    "Return ONLY valid JSON that matches the schema exactly."
                )

            messages = [
                {
                    "role": "system",
                    "content": "You output strict JSON only. No markdown. No extra keys beyond the schema.",
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        f"Placeholders you must use:\n"
                        f'- user_id: "{{user_id}}"\n'
                        f'- candidate_list: "{{candidate_list}}"\n\n'
                        f"Current values (do NOT copy these into the plan directly):\n"
                        f"user_id={user_id}\n"
                        f"candidate_list_len={len(candidate_list)}\n"
                        f"{retry_suffix}"
                    ),
                },
            ]

            llm_out = self.llm(
                messages=messages,
                temperature=0.0,
                max_tokens=600,
                # If your LLM wrapper supports structured output, keep it:
                response_format={"type": "json_object"},
            )

            plan_obj, parse_err = _try_json_loads(llm_out if isinstance(llm_out, str) else str(llm_out))
            if parse_err:
                last_errs = [f"JSON parse error: {parse_err}"]
                continue

            valid_plan, errs = _validate_plan_obj(plan_obj)
            if valid_plan is not None:
                return valid_plan

            last_errs = errs

        # Fallback
        # Optionally log last_errs somewhere
        return copy.deepcopy(DEFAULT_FALLBACK_PLAN)

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm, tools):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        self.tools = tools
        
    def __call__(self, user, items, task_description: str, plan):
        """
        Execute a TYPED plan produced by RecPlanning, without English substring routing.

        plan can be:
        - a dict: {"steps":[...]}
        - a list: [...]
        - a JSON string containing either of the above
        Each step should look like:
        {"type":"FETCH_USER", "params":{"user_id":"{{user_id}}"}, ...}
        """

        # 1) Normalize plan into a list of typed steps
        if isinstance(plan, str):
            plan = json.loads(plan)

        if isinstance(plan, dict) and "steps" in plan:
            steps = plan["steps"]
        elif isinstance(plan, list):
            steps = plan
        else:
            steps = []

        # 2) Fill placeholders
        def fill_placeholders(obj):
            if isinstance(obj, str):
                if obj == "{{user_id}}":
                    return user
                if obj == "{{candidate_list}}":
                    return items
                return obj
            if isinstance(obj, list):
                return [fill_placeholders(x) for x in obj]
            if isinstance(obj, dict):
                return {k: fill_placeholders(v) for k, v in obj.items()}
            return obj

        steps = fill_placeholders(steps)

        # 3) Execute steps deterministically using tools
        reasoning_process = []  # list of executed step records (stable, machine-readable)

        for step in steps:
            step_type = (step.get("type") or "").upper()
            params = step.get("params") or {}
            record = {
                "type": step_type,
                "params": params,
                "tool": None,
                "tool_input": None,
                "tool_output": None,
            }

            # Map step type -> tool calls
            if step_type == "FETCH_USER":
                record["tool"] = "get_user"
                record["tool_input"] = {"user_id": params.get("user_id", user)}
                record["tool_output"] = self.tools["get_user"]["function"](**record["tool_input"])

            elif step_type == "FETCH_ITEMS":
                item_ids = params.get("item_ids", items)
                if "get_items" in self.tools:
                    record["tool"] = "get_items"
                    record["tool_input"] = {"item_ids": item_ids}
                    record["tool_output"] = self.tools["get_items"]["function"](**record["tool_input"])
                else:
                    record["tool"] = "get_item"
                    record["tool_input"] = {"item_id": "<loop>"}
                    record["tool_output"] = [
                        self.tools["get_item"]["function"](item_id=iid) for iid in item_ids
                    ]

            elif step_type == "FETCH_USER_REVIEWS":
                record["tool"] = "get_reviews"
                record["tool_input"] = {"user_id": params.get("user_id", user)}
                record["tool_output"] = self.tools["get_reviews"]["function"](**record["tool_input"])

            elif step_type == "FETCH_CANDIDATE_REVIEWS":
                item_ids = params.get("item_ids", items)
                record["tool"] = "get_reviews"
                record["tool_input"] = {"item_id": "<loop>"}
                record["tool_output"] = {
                    iid: self.tools["get_reviews"]["function"](item_id=iid) for iid in item_ids
                }

            elif step_type == "RANK":
                # No tool call; ranking happens after evidence is collected
                pass

            else:
                record["tool_output"] = {"warning": f"Unknown step type: {step_type}"}

            reasoning_process.append(record)
            print(f"Executed step: {step_type} -> tool={record['tool']}")

        # 4) One final LLM call to rank using the gathered evidence
        reasoning_result = self.llm(
            messages=[
                {
                    "role": "assistant",
                    "content": (json.dumps(reasoning_process))
                },
                {
                    "role": "system",
                    "content": (
                        "You are a recommendation agent. Consider the user's preferences and how they relate to the candidate items given to rank the candidate items." 
                        "Output ONLY strict JSON:\n"
                        "{\"ranked_ids\": [\"id1\", \"id2\", ...]}\n"
                        "ranked_ids must contain ONLY the provided candidate item ids, each exactly once."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{task_description}\n\n"
                        f"User: {user}\n"
                        f"Candidate items: {items}\n\n"
                        f"Executed typed-step evidence:\n{json.dumps(reasoning_process)}\n\n"
                        "Return ranked_ids only."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=2048,
        )

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
    """
    Refactor-aligned agent:
      1) Uses RecPlanning.plan_with_validation(...) to get a typed plan (or fallback)
      2) Does NOT do English substring routing or manual tool calls in workflow()
      3) Uses RecReasoning to deterministically execute typed steps + do one final LLM ranking
      4) Parses final ranker output as JSON and returns ranked_ids robustly
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.reasoning: Optional[RecReasoning] = None

    def set_interaction_tool(self, interaction_tool):
        super().set_interaction_tool(interaction_tool)

        # Optional: wrap tool calls to keep evidence size manageable for the final LLM prompt.
        # These wrappers are defensive. They avoid huge payloads that can blow the context window.
        def _truncate_text(x: Any, max_chars: int = 800) -> Any:
            if isinstance(x, str) and len(x) > max_chars:
                return x[:max_chars] + "..."
            return x

        def _compact_item(item: Any) -> Any:
            if not isinstance(item, dict):
                return item
            keep = [
                "item_id", "name", "title", "stars", "average_rating",
                "review_count", "rating_number", "ratings_count",
                "attributes", "categories", "description", "title_without_series",
            ]
            out = {k: item.get(k) for k in keep if k in item}
            if "description" in out:
                out["description"] = _truncate_text(out["description"], 800)
            return out

        def _compact_review(r: Any) -> Any:
            if not isinstance(r, dict):
                return r
            keep = ["review_id", "user_id", "item_id", "stars", "rating", "text", "time", "date"]
            out = {k: r.get(k) for k in keep if k in r}
            if "text" in out:
                out["text"] = _truncate_text(out["text"], 800)
            return out

        def get_user_wrapped(user_id: str):
            u = self.interaction_tool.get_user(user_id=user_id)
            if isinstance(u, dict):
                # Keep most relevant fields but do not overprune
                for k in list(u.keys()):
                    u[k] = _truncate_text(u[k], 800)
            return u

        def get_item_wrapped(item_id: str):
            return _compact_item(self.interaction_tool.get_item(item_id=item_id))

        def get_items_wrapped(item_ids: List[str]):
            items = self.interaction_tool.get_items(item_ids=item_ids)
            if isinstance(items, list):
                return [_compact_item(it) for it in items]
            return items

        def get_reviews_wrapped(item_id: Optional[str] = None, user_id: Optional[str] = None, review_id: Optional[str] = None):
            reviews = self.interaction_tool.get_reviews(item_id=item_id, user_id=user_id, review_id=review_id)
            # Cap and compact for prompt size safety
            if isinstance(reviews, list):
                reviews = reviews[:50]
                return [_compact_review(r) for r in reviews]
            return reviews

        self.tools = {
            "get_user": {
                "function": get_user_wrapped,
                "description": "Fetch user data based on user_id",
                "parameters": {"user_id": "str"},
            },
            "get_item": {
                "function": get_item_wrapped,
                "description": "Fetch item data based on item_id",
                "parameters": {"item_id": "str"},
            },
            "get_items": {
                "function": get_items_wrapped,
                "description": "Fetch multiple items based on a list of item_ids",
                "parameters": {"item_ids": "List[str]"},
            },
            "get_reviews": {
                "function": get_reviews_wrapped,
                "description": "Fetch reviews filtered by various parameters",
                "parameters": {"item_id": "Optional[str]", "user_id": "Optional[str]", "review_id": "Optional[str]"},
            },
        }

        self.reasoning = RecReasoning(profile_type_prompt="", llm=self.llm, tools=self.tools)

    def workflow(self) -> List[str]:
        if self.reasoning is None:
            raise RuntimeError("Reasoning module not initialized. set_interaction_tool was not called.")

        user_id: str = self.task["user_id"]
        candidate_list: List[str] = self.task["candidate_list"]

        # 1) Produce a typed plan with validation (or fallback)
        plan_task_description = (
            "Rank the candidate items for the user. You may need user profile, user review history, "
            "candidate item metadata, and candidate item reviews. End with a RANK step."
        )

        plan = self.planning.plan_with_validation(
            task_type="Recommendation Task",
            task_description=plan_task_description,
            user_id=user_id,
            candidate_list=candidate_list,
            feedback="",
            few_shot="",
            max_attempts=3,
        )

        # 2) No manual substring routing, no manual tool calls
        # 3) Task description for the final ranker (no tool-use instructions here)
        reasoning_task_description = (
            "You will be given evidence from tool calls (user, items, user reviews, candidate reviews). "
            "Use that evidence to rank candidates based on predicted preference match. Prefer items aligned "
            "with the user's positive themes, avoid items matching negative themes, and break ties by stronger "
            "quality signals (ratings, review sentiment) when available."
        )

        # Run deterministic typed-step execution + final LLM ranking
        llm_out = self.reasoning(
            user=user_id,
            items=candidate_list,
            plan=plan,
            task_description=reasoning_task_description,
        )

        # 4) Parse strict JSON output: {"ranked_ids": [...]}
        obj, err = _try_json_loads(llm_out if isinstance(llm_out, str) else str(llm_out))
        ranked_ids: List[str] = []

        if isinstance(obj, dict) and isinstance(obj.get("ranked_ids"), list):
            ranked_ids = [x for x in obj["ranked_ids"] if isinstance(x, str)]
        elif isinstance(obj, list):
            # tolerate list-only outputs
            ranked_ids = [x for x in obj if isinstance(x, str)]

        # Repair to ensure: only candidates, each once, no missing
        cand_set = set(candidate_list)
        seen = set()
        cleaned: List[str] = []
        for iid in ranked_ids:
            if iid in cand_set and iid not in seen:
                cleaned.append(iid)
                seen.add(iid)

        # Append any missing candidates deterministically
        if len(cleaned) < len(candidate_list):
            for iid in candidate_list:
                if iid not in seen:
                    cleaned.append(iid)
                    seen.add(iid)

        # Final safety fallback
        if not cleaned:
            cleaned = copy.deepcopy(candidate_list)

        return cleaned
        
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
    simulator.set_llm(OllamaLLM(model="qwen2.5"))

    # Run evaluation
    # If you don't set the number of tasks, the simulator will run all tasks.
    agent_outputs = simulator.run_simulation(number_of_tasks=1, enable_threading=True, max_workers=10)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track2_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")