"""
Design2Code Evaluator - Green agent that runs Design2Code evaluation on white agents.

This agent:
1. Loads Design2Code dataset from Hugging Face
2. Sends screenshot tasks to the white agent (the agent being tested)
3. Parses HTML from the white agent's response
4. Evaluates the generated HTML and collects metrics
"""
import argparse
import asyncio
import base64
import io
import json
import logging
import re
import time
from typing import Any, Optional

import uvicorn
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest
from agentbeats.tool_provider import ToolProvider

# Import evaluation module - adjust path to handle script execution
import sys
from pathlib import Path
evaluation_path = Path(__file__).parent / "evaluation"
sys.path.insert(0, str(Path(__file__).parent))
from evaluation.visual_evaluator import evaluate_html

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("design2code_evaluator")


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_html_from_response(response: str) -> str:
    """Extract HTML code from <html_code>...</html_code> tags."""
    match = re.search(r'<html_code>\s*(.*?)\s*</html_code>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no tags found, return the full response
    return response.strip()


def is_refusal_response(response: str) -> bool:
    """Check if the LLM response is a refusal to assist."""
    refusal_patterns = [
        r"i\s+can'?t\s+assist",
        r"i\s+can'?t\s+help",
        r"i'?m\s+sorry,\s+i\s+can'?t",
        r"i\s+can'?t\s+provide",
        r"i\s+am\s+not\s+able\s+to",
        r"i\s+don'?t\s+have\s+the\s+ability",
        r"i\s+can'?t\s+generate",
        r"unable\s+to\s+assist",
        r"cannot\s+assist",
        r"cannot\s+help",
    ]
    response_lower = response.lower()
    for pattern in refusal_patterns:
        if re.search(pattern, response_lower):
            return True
    return False


def get_task_objects(
    dataset: Any,
    task_ids: Optional[list[str | int]],
    num_tasks: Optional[int] = None,
) -> list[tuple[int, dict]]:
    """Get task objects (index, item) from the dataset, optionally limited to num_tasks."""
    try:
        if task_ids is not None:
            # Filter by specific task IDs
            # Convert task_ids to indices if they're integers
            if all(isinstance(tid, int) for tid in task_ids):
                indices = task_ids
            else:
                # If task_ids are strings, try to find them in the dataset
                indices = [i for i, item in enumerate(dataset) if str(item.get("id", i)) in [str(tid) for tid in task_ids]]

            if not indices:
                # If no matches, use task_ids as indices
                indices = [int(tid) if isinstance(tid, str) and tid.isdigit() else int(tid) for tid in task_ids]

            result = [(i, dataset[i]) for i in indices]
        else:
            # Use all tasks, limited by num_tasks
            total = len(dataset)
            limit = num_tasks if num_tasks is not None else total
            result = [(i, dataset[i]) for i in range(min(limit, total))]

        return result
    except Exception as e:
        logger.error(f"Error getting task objects: {e}")
        # Fallback: return default task objects if dataset access fails
        limit = num_tasks if num_tasks is not None else 3
        return [(i, dataset[i]) for i in range(min(limit, len(dataset)))]


class Design2CodeEvaluator(GreenAgent):
    """Green agent that evaluates white agents using Design2Code benchmark."""

    def __init__(self):
        self._required_roles = ["agent"]  # The white agent being tested
        self._required_config_keys = ["dataset_name"]
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting Design2Code evaluation: {req}")
        start_time = time.time()

        dataset_name = req.config["dataset_name"]
        task_ids = req.config.get("task_ids", None)
        num_tasks = req.config.get("num_tasks", None)

        # Get the white agent URL
        agent_url = str(req.participants["agent"])

        # Load dataset
        try:
            dataset = load_dataset(dataset_name, split="train")
            logger.info(f"Loaded dataset: {dataset_name}, {len(dataset)} tasks")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            # Raise exception to let GreenExecutor handle error reporting
            raise RuntimeError(f"Failed to load dataset: {e}") from e

        # Get task objects
        tasks = get_task_objects(dataset, task_ids, num_tasks)
        logger.info(f"Running {len(tasks)} tasks from {dataset_name}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(tasks)} tasks from {dataset_name}")
        )

        metrics: dict[str, Any] = {"tasks": {}}

        try:
            for task_idx, task_data in tasks:
                logger.info(f"Running task {task_idx}...")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task {task_idx}...")
                )

                try:
                    reward = await self._run_single_task(
                        agent_url=agent_url,
                        task_data=task_data,
                        task_idx=task_idx,
                    )
                    metrics["tasks"][str(task_idx)] = reward
                    logger.info(f"Task {task_idx} completed with reward: {reward}")
                except Exception as e:
                    logger.error(f"Task {task_idx} failed: {e}", exc_info=True)
                    metrics["tasks"][str(task_idx)] = 0.0

            time_used = time.time() - start_time
            total_reward = sum(metrics["tasks"].values())
            num_completed = len(metrics["tasks"])
            avg_reward = total_reward / num_completed if num_completed > 0 else 0.0
            pass_rate = (sum(1 for r in metrics["tasks"].values() if r > 0.5) / num_completed * 100) if num_completed > 0 else 0

            result_data = {
                "dataset_name": dataset_name,
                "avg_score": avg_reward,
                "total_score": total_reward,
                "num_tasks": num_completed,
                "pass_rate": pass_rate,
                "task_scores": metrics["tasks"],
                "time_used": time_used,
            }

            # Format task results for display
            task_results_str = "\n".join(
                f"  Task {task_id}: {'✅' if score > 0.7 else '⚠️' if score > 0.5 else '❌'} ({score:.3f})"
                for task_id, score in sorted(metrics["tasks"].items(), key=lambda x: int(x[0]))
            )

            summary = f"""Design2Code Benchmark Results
Dataset: {dataset_name}
Tasks: {num_completed}
Average Score: {avg_reward:.3f}
Pass Rate: {pass_rate:.1f}% ({int(total_reward)}/{num_completed})
Time: {time_used:.1f}s

Task Results:
{task_results_str}"""

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result_data)),
                ],
                name="Result",
            )

        finally:
            self._tool_provider.reset()

    async def _run_single_task(
        self,
        agent_url: str,
        task_data: dict,
        task_idx: int,
    ) -> float:
        """Run a single Design2Code task and return the reward."""
        try:
            # Log dataset structure for debugging
            if task_idx == 0:  # Only log for first task to avoid spam
                if hasattr(task_data, 'keys'):
                    logger.info(f"Dataset fields (from first task): {list(task_data.keys())}")
                elif isinstance(task_data, dict):
                    logger.info(f"Dataset fields (from first task): {list(task_data.keys())}")
                else:
                    logger.info(f"Task data type: {type(task_data)}, has keys: {hasattr(task_data, 'keys')}")

            # Extract screenshot (should be a PIL Image)
            screenshot = task_data.get("image") or task_data.get("screenshot")
            if screenshot is None:
                logger.warning(f"Task {task_idx} has no screenshot/image")
                return 0.0

            # Ensure it's a PIL Image
            if not isinstance(screenshot, Image.Image):
                try:
                    # Try to convert to PIL Image
                    screenshot = Image.fromarray(screenshot)
                except (TypeError, ValueError):
                    # If that fails, try to use it directly if it has convert method
                    if hasattr(screenshot, 'convert'):
                        screenshot = screenshot.convert('RGB')
                    else:
                        logger.warning(f"Task {task_idx}: Could not convert screenshot to PIL Image")
                        return 0.0

            # Convert to RGB if needed
            if screenshot.mode != 'RGB':
                screenshot = screenshot.convert('RGB')

            # Get reference HTML - try multiple possible field names
            # For Design2Code dataset, the 'text' field contains the reference HTML
            reference_html = (
                task_data.get("text")
            )

            if reference_html:
                logger.debug(f"Task {task_idx}: Found reference HTML (length: {len(reference_html)})")
            else:
                logger.warning(f"Task {task_idx}: No reference HTML found in any field")

            # Convert screenshot to base64
            screenshot_base64 = image_to_base64(screenshot)

            # Build task prompt - just include the screenshot, let SYSTEM_PROMPT handle instructions
            task_prompt = f"""<screenshot_base64>
{screenshot_base64}
</screenshot_base64>
"""

            # Send to white agent with retry logic for refusal responses
            max_retries = 3
            retry_delay = 1.0  # seconds
            response = None
            generated_html = None

            for attempt in range(max_retries):
                logger.debug(f"Sending task {task_idx} to white agent (attempt {attempt + 1}/{max_retries})...")
                response = await self._tool_provider.talk_to_agent(
                    message=task_prompt,
                    url=agent_url,
                    new_conversation=(attempt == 0),  # Only start new conversation on first attempt
                )

                logger.debug(f"White agent response (first 200 chars): {response[:200]}...")

                # Check if response is a refusal
                if is_refusal_response(response):
                    logger.warning(f"Task {task_idx}: LLM refused to assist (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying task {task_idx} in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Task {task_idx}: LLM refused after {max_retries} attempts")
                        return 0.0

                # Extract HTML from response
                generated_html = extract_html_from_response(response)

                # If we got HTML content, break out of retry loop
                if generated_html and len(generated_html.strip()) > 0:
                    break
                else:
                    logger.warning(f"Task {task_idx}: Empty HTML extracted (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying task {task_idx} in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue

            if not generated_html or len(generated_html.strip()) == 0:
                logger.warning(f"Task {task_idx}: No HTML extracted from response after {max_retries} attempts")
                return 0.0

            # Evaluate HTML using visual evaluator
            score = await evaluate_html(
                generated_html=generated_html,
                reference_html=reference_html,
                reference_image=screenshot,
            )

            return float(score)

        except Exception as e:
            logger.error(f"Error running task {task_idx}: {e}", exc_info=True)
            raise


def design2code_evaluator_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the design2code evaluator."""
    skill = AgentSkill(
        id="design2code_evaluation",
        name="Design2Code Benchmark Evaluation",
        description="Evaluates agents on HTML generation from screenshots",
        tags=["benchmark", "evaluation", "design2code"],
        examples=[
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"dataset_name": "SALT-NLP/Design2Code-hf", "num_tasks": 5}}'
        ],
    )
    return AgentCard(
        name=name,
        description="Design2Code benchmark evaluator - tests agents on HTML generation from screenshots",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def main():
    parser = argparse.ArgumentParser(description="Run the design2code evaluator agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    agent = Design2CodeEvaluator()
    executor = GreenExecutor(agent)
    agent_card = design2code_evaluator_agent_card("Design2CodeEvaluator", agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
