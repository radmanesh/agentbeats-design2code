"""
Tau2 Evaluator - Green agent that runs tau-bench evaluation on purple agents.

This agent uses tau2's native Orchestrator for evaluation. The purple agent
being tested is wrapped in a RemoteA2AAgent that communicates via A2A protocol.
"""
import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Any, List, Optional

import nest_asyncio
import uvicorn
from dotenv import load_dotenv

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

from tau2.agent.base import BaseAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tau2_evaluator")

# Allow nested event loops (needed for sync/async bridge in RemoteA2AAgent)
nest_asyncio.apply()

RESPOND_ACTION_NAME = "respond"


def tools_to_str(tools: List[Tool]) -> str:
    """Convert tau-bench tools to JSON schema format."""
    return json.dumps([tool.openai_schema for tool in tools], indent=2)


def get_task_objects(domain: str, task_ids: Optional[List[str]], num_tasks: Optional[int] = None):
    """Get task objects for the domain, optionally limited to num_tasks."""
    task_set_name = domain
    task_split_name = "base"
    if task_ids is None:
        tasks = get_tasks(task_set_name=task_set_name, task_split_name=task_split_name)
    else:
        tasks = get_tasks(
            task_set_name=task_set_name,
            task_split_name=task_split_name,
            task_ids=task_ids,
        )

    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def extract_text_from_message(message: MultiToolMessage | UserMessage | ToolMessage) -> str | None:
    # Build the message to send to remote agent
    if isinstance(message, UserMessage):
        outgoing_text = message.content
    elif isinstance(message, MultiToolMessage):
        # Format tool results
        tool_results = []
        for tm in message.tool_messages:
            tool_results.append(f"Tool '{tm.name}' result: {tm.content}")
        outgoing_text = "\n".join(tool_results)
    else:
        outgoing_text = str(message.content) if hasattr(message, 'content') else str(message)
    return outgoing_text


class RemoteA2AAgent(BaseAgent):
    """
    An agent that delegates to a remote purple agent via A2A protocol.

    This implements tau2's BaseAgent interface so it can be used with
    the native Orchestrator, while delegating actual decision-making
    to the remote agent being tested.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        tool_provider: ToolProvider,
        agent_url: str,
    ):
        self.tools = tools
        self.domain_policy = domain_policy
        self.tool_provider = tool_provider
        self.agent_url = agent_url
        self._is_first_message = True

    @property
    def agent_prompt(self) -> str:
        """Build the system prompt with policy and tools."""
        return f"""{self.domain_policy}

Here's a list of tools you can use (you can use at most one tool at a time):
{tools_to_str(self.tools)}

and 

{json.dumps({
    "type": "function",
    "function": {
        "name": RESPOND_ACTION_NAME,
        "description": "Respond directly to the user with a message instead of calling a tool.",
        "parameters": {
            "properties": {
                "content": {
                    "description": "The message content to send to the user.",
                    "title": "Content",
                    "type": "string"
                }
            },
            "required": ["content"],
            "title": "parameters",
            "type": "object"
        }
    }
}, indent=2)}


Please respond in JSON format.
The JSON should contain:
- "name": the tool call function name.
- "arguments": the arguments for the tool call.

You should only use one tool at a time!
You cannot respond to user and use a tool at the same time!

Examples of responses:
<json>
{json.dumps({"name": "find_user_id_by_name_zip", "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip_code": "19122"}}, indent=2)}
</json>

<json>
{json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content": "Hello, how can I help you today?"}}, indent=2)}
</json>
"""

    def get_init_state(self, message_history: Optional[list] = None) -> LLMAgentState:
        """Get the initial state of the agent."""
        if message_history is None:
            message_history = []
        self._is_first_message = True
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.agent_prompt)],
            messages=message_history,
        )

    def set_seed(self, seed: int):
        """Set random seed (no-op for remote agent)."""
        pass

    def stop(self, last_message=None, state=None):
        """Stop the agent (no-op for remote agent)."""
        pass

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Generate the next message by delegating to the remote purple agent.

        This method is synchronous (as required by tau2), but internally
        uses asyncio to communicate with the remote agent.
        """
        # Update state with incoming message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        outgoing_text = extract_text_from_message(message)

        # If first message, prepend system prompt and all messages.
        if self._is_first_message:
            outgoing_text = f"{self.agent_prompt}\n\nNow here are the user messages:\n{'\n'.join([extract_text_from_message(message) for message in state.messages])}"

        # Call remote agent via A2A
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(
            self.tool_provider.talk_to_agent(
                message=outgoing_text,
                url=self.agent_url,
                new_conversation=self._is_first_message,
            )
        )
        self._is_first_message = False

        # Parse the response
        assistant_message = self._parse_response(response)
        state.messages.append(assistant_message)

        return assistant_message, state

    def _parse_response(self, response: str) -> AssistantMessage:
        """Parse the purple agent's response into an AssistantMessage."""
        try:
            action_dict = json.loads(response)

            is_tool_call = action_dict["name"] != RESPOND_ACTION_NAME

            if not is_tool_call:
                # Response to user
                return AssistantMessage(
                    role="assistant",
                    content=action_dict["arguments"]["content"],
                    tool_calls=None,
                )
            else:
                # Tool call
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=action_dict["name"],
                    arguments=action_dict["arguments"],
                    requestor="assistant",
                )
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                )
        except (json.JSONDecodeError, KeyError) as e:
            # If parsing fails, treat the response as plain text to user
            logger.warning(f"Failed to parse agent response as JSON: {e}")
            return AssistantMessage(
                role="assistant",
                content=response,
                tool_calls=None,
            )


class Tau2Evaluator(GreenAgent):
    """Green agent that evaluates purple agents using tau2's native Orchestrator."""

    def __init__(self):
        self._required_roles = ["agent"]  # The purple agent being tested
        self._required_config_keys = ["domain"]
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
        logger.info(f"Starting tau2 evaluation: {req}")
        start_time = time.time()

        domain = req.config["domain"]
        task_ids = req.config.get("task_ids", None)
        num_tasks = req.config.get("num_tasks", None)
        max_steps = req.config.get("max_steps", 200)
        user_llm = req.config.get("user_llm", "openai/gpt-4.1")
        user_llm_args = req.config.get("user_llm_args", {"temperature": 0.0})

        # Get the purple agent URL
        agent_url = str(req.participants["agent"])

        # Get task objects
        tasks = get_task_objects(domain, task_ids, num_tasks)
        logger.info(f"Running {len(tasks)} tasks for domain {domain}")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(tasks)} tasks in {domain} domain")
        )

        metrics: dict[str, Any] = {"tasks": {}}

        try:
            for task in tasks:
                task_id = task.id
                logger.info(f"Running task {task_id}...")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task {task_id}...")
                )

                try:
                    reward = await self._run_single_task(
                        agent_url=agent_url,
                        domain=domain,
                        task=task,
                        max_steps=max_steps,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                    )
                    metrics["tasks"][task_id] = reward
                    logger.info(f"Task {task_id} completed with reward: {reward}")
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                    metrics["tasks"][task_id] = 0.0

            time_used = time.time() - start_time
            total_reward = sum(metrics["tasks"].values())
            num_completed = len(metrics["tasks"])
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0

            result_data = {
                "domain": domain,
                "score": total_reward,
                "max_score": num_completed,
                "pass_rate": pass_rate,
                "task_rewards": metrics["tasks"],
                "time_used": time_used,
            }

            # Format task results for display
            task_results_str = "\n".join(
                f"  {task_id}: {'✓' if reward == 1.0 else '✗'} ({reward})"
                for task_id, reward in metrics["tasks"].items()
            )

            summary = f"""Tau2 Benchmark Results
Domain: {domain}
Tasks: {num_completed}
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
        domain: str,
        task,
        max_steps: int,
        user_llm: str,
        user_llm_args: dict,
    ) -> float:
        """Run a single tau-bench task using native Orchestrator and return the reward."""

        # Get environment from registry
        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor(solo_mode=False)

        # Create the remote agent wrapper
        agent = RemoteA2AAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            tool_provider=self._tool_provider,
            agent_url=agent_url,
        )

        # Create user simulator
        user = UserSimulator(
            tools=environment.get_user_tools() if environment.user_tools else None,
            instructions=str(task.user_scenario),
            llm=user_llm,
            llm_args=user_llm_args,
        )

        # Create orchestrator
        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=10,
            seed=42,
            solo_mode=False,
            validate_communication=False,
        )

        # Run the simulation
        simulation_run = orchestrator.run()

        logger.info(f"Task {task.id} terminated: {simulation_run.termination_reason}")
        logger.debug(f"Task {task.id} messages: {len(simulation_run.messages)}")

        # Evaluate the simulation
        try:
            reward_info = evaluate_simulation(
                simulation=simulation_run,
                task=task,
                evaluation_type=EvaluationType.ACTION,
                solo_mode=False,
                domain=domain,
            )
            return reward_info.reward
        except Exception as e:
            logger.error(f"Evaluation failed for task {task.id}: {e}")
            return 0.0


def tau2_evaluator_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the tau2 evaluator."""
    skill = AgentSkill(
        id="tau2_evaluation",
        name="Tau2 Benchmark Evaluation",
        description="Evaluates agents on tau-bench tasks (airline, retail domains)",
        tags=["benchmark", "evaluation", "tau2"],
        examples=[
            '{"participants": {"agent": "http://localhost:9019"}, "config": {"domain": "airline", "num_tasks": 5}}'
        ],
    )
    return AgentCard(
        name=name,
        description="Tau2 benchmark evaluator - tests agents on customer service tasks",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def main():
    parser = argparse.ArgumentParser(description="Run the tau2 evaluator agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    agent = Tau2Evaluator()
    executor = GreenExecutor(agent)
    agent_card = tau2_evaluator_agent_card("Tau2Evaluator", agent_url)

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
