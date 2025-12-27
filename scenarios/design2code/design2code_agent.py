"""
Design2Code Agent - White agent that generates HTML from screenshots.

This is the agent being tested. It:
1. Receives screenshot images and task instructions
2. Uses GPT-4o Vision to generate HTML code that recreates the visual appearance
3. Returns HTML wrapped in <html_code>...</html_code> tags
"""
import argparse
import base64
import re
import sys
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from litellm import completion
from loguru import logger

def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the design2code agent."""
    skill = AgentSkill(
        id="html_generation",
        name="HTML Generation",
        description="Generates HTML code from screenshot images for Design2Code evaluation",
        tags=["benchmark", "design2code"],
        examples=[],
    )
    return AgentCard(
        name="design2code_agent",
        description="HTML generation agent for Design2Code evaluation",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


SYSTEM_PROMPT = """You are an expert web developer who specializes in HTML and CSS.
A user will provide you with a screenshot of a webpage.
You need to return a single html file that uses HTML and CSS to reproduce the given website.
Include all CSS code in the HTML file itself.
If it involves any images, use "rick.jpg" as the placeholder.
Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well.
Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.
Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.
Respond with the content of the HTML+CSS file wrapped in <html_code>...</html_code> tags."""


class Design2CodeAgentExecutor(AgentExecutor):
    """Executor for the design2code agent."""

    def __init__(self):
        self.ctx_id_to_messages: dict[str, list[dict]] = {}

    def _extract_screenshot(self, user_input: str) -> tuple[str, str | None]:
        """Extract screenshot base64 from input and return remaining text and image."""
        # Try to extract screenshot from <screenshot_base64>...</screenshot_base64> tags
        match = re.search(r'<screenshot_base64>\s*(.*?)\s*</screenshot_base64>', user_input, re.DOTALL)
        if match:
            base64_image = match.group(1).strip()
            # Remove the screenshot tags from the input
            remaining_text = re.sub(r'<screenshot_base64>.*?</screenshot_base64>', '', user_input, flags=re.DOTALL).strip()
            return remaining_text, base64_image
        return user_input, None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        logger.info(f"Received input: {user_input[:200]}...")

        # Extract screenshot and remaining text
        text_content, screenshot_base64 = self._extract_screenshot(user_input)

        # Initialize or get conversation history
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        messages = self.ctx_id_to_messages[context.context_id].copy()

        # Ensure system prompt is always first in messages
        has_system_prompt = any(msg.get("role") == "system" for msg in messages)
        if not has_system_prompt:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            # Also ensure it's in stored messages
            if not any(msg.get("role") == "system" for msg in self.ctx_id_to_messages[context.context_id]):
                self.ctx_id_to_messages[context.context_id].insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        # Build user message with image if present
        user_message = None
        if screenshot_base64:
            # Add image as data URL - use list format for vision models
            image_url = f"data:image/png;base64,{screenshot_base64}"
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
            if text_content:
                content.append({
                    "type": "text",
                    "text": text_content
                })
            user_message = {
                "role": "user",
                "content": content
            }
            messages.append(user_message)
        else:
            # No image, use simple text format
            user_message = {
                "role": "user",
                "content": text_content or "Generate HTML code."
            }
            messages.append(user_message)

        # Add user message to stored conversation history
        if user_message:
            self.ctx_id_to_messages[context.context_id].append(user_message)

        # Ensure system prompt is still in messages before LLM call
        final_has_system = any(msg.get("role") == "system" for msg in messages)
        if not final_has_system:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        # Call LLM with vision support
        try:
            response = completion(
                messages=messages,
                model="openai/gpt-4o",
                temperature=0.0,
            )
            assistant_content = response.choices[0].message.content
            logger.info(f"LLM response: {assistant_content[:200]}...")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            assistant_content = '<html_code>\n<!-- Error: Failed to generate HTML -->\n</html_code>'

        # Add assistant response to history
        # Ensure system prompt is still in stored messages
        stored_messages = self.ctx_id_to_messages[context.context_id]
        if not any(msg.get("role") == "system" for msg in stored_messages):
            stored_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        stored_messages.append({
            "role": "assistant",
            "content": assistant_content
        })

        # Ensure HTML is wrapped in tags
        if "<html_code>" not in assistant_content:
            assistant_content = f"<html_code>\n{assistant_content}\n</html_code>"

        # Send response back via A2A
        await event_queue.enqueue_event(
            new_agent_text_message(assistant_content, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Run the design2code agent (white agent).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    args = parser.parse_args()

    logger.info("Starting design2code agent...")
    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=Design2CodeAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
