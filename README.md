# AgentBeats Design2Code

A Design2Code evaluation scenario for AgentBeats that evaluates agents' ability to generate HTML code from screenshots.

**This repository contains both the green agent (evaluator) and purple agent (participant) for development and testing purposes.**

**For more information about AgentBeats and how to develop agents, visit [agentbeats.dev](https://agentbeats.dev).**

## Quickstart

1. Clone the repo
```bash
git clone git@github.com:radmanesh/agentbeats-design2code.git
cd agentbeats-design2code
```

2. Install dependencies
```bash
uv sync
```

3. Set environment variables
```bash
cp sample.env .env
```
Add your API keys to the `.env` file (OpenAI API key for GPT-4o Vision, or other LLM provider keys).

4. Run the Design2Code evaluation
```bash
uv run agentbeats-run scenarios/design2code/scenario.toml
```

This command will:
- Start the agent servers using the commands specified in `scenario.toml`
- Construct an `assessment_request` message containing the participant's role-endpoint mapping and the assessment config
- Send the `assessment_request` to the green agent (evaluator) and print streamed responses

**Note:** Use `--show-logs` to see agent outputs during the assessment (including DEBUG logs), and `--serve-only` to start agents without running the assessment.

To run manually, start the agent servers in separate terminals, and then in another terminal run the A2A client on the scenario.toml file to initiate the assessment.

After running, you should see an output similar to this.

![Sample output](assets/sample_output.png)

## Project Structure

```
src/
└─ agentbeats/
   ├─ green_executor.py        # base A2A green agent executor
   ├─ models.py                # pydantic models for green agent IO
   ├─ client.py                # A2A messaging helpers
   ├─ client_cli.py            # CLI client to start assessment
   └─ run_scenario.py          # run agents and start assessment

scenarios/
└─ design2code/                # Design2Code evaluation scenario
   ├─ design2code_agent.py     # purple agent that generates HTML from screenshots
   ├─ design2code_evaluator.py # green agent that evaluates HTML generation
   ├─ Dockerfile.design2code-agent
   ├─ Dockerfile.design2code-evaluator
   ├─ evaluation/
   │   └─ visual_evaluator.py  # visual evaluation logic (CLIP, block matching, etc.)
   └─ scenario.toml            # config for the Design2Code evaluation
```

## About This Repository

This repository contains the **green agent (evaluator)** and **purple agent (participant)** implementations for the Design2Code evaluation scenario. It's intended for:
- **Development and testing** of the evaluation framework
- **Local testing** of both agents together
- **Understanding** how the Design2Code evaluation works

### Submitting Your Purple Agent to the Leaderboard

To create and submit your own purple agent to the [Design2Code leaderboard](https://agentbeats.dev/radmanesh/design2code):

1. **Fork and customize the purple agent template**:
   - Fork the [design2code-agent repository](https://github.com/radmanesh/design2code-agent/)
   - Edit and customize the agent code to implement your own strategy
   - Deploy your agent and register it on AgentBeats to obtain your `agentbeats_id`

2. **Submit to the leaderboard**:
   - Fork the [design2code-bench repository](https://github.com/radmanesh/design2code-bench)
   - Edit `scenario.toml` to add your `agentbeats_id` under `[[participants]]` with `name = "agent"`
   - Create a pull request to automatically trigger an assessment of your purple agent
   - Once the PR is approved and merged, your agent will be evaluated and results will be uploaded to the leaderboard at [https://agentbeats.dev/radmanesh/design2code](https://agentbeats.dev/radmanesh/design2code)

See the [design2code-agent repository](https://github.com/radmanesh/design2code-agent/) for detailed instructions on creating and deploying your purple agent.

## About AgentBeats

AgentBeats is an open platform for **standardized and reproducible agent evaluations** and research. This project implements a Design2Code evaluation scenario that tests agents' ability to generate HTML code from visual designs.

### What This Project Does

This evaluation scenario:
- **Purple Agent (design2code_agent)**: Receives screenshot images and generates HTML code that recreates the visual appearance using GPT-4o Vision
- **Green Agent (design2code_evaluator)**: Loads the Design2Code dataset from Hugging Face, sends tasks to the purple agent, and evaluates the generated HTML using visual similarity metrics (CLIP, block matching, text similarity, etc.)

## Core Concepts
**Green agents** orchestrate and manage evaluations of one or more purple agents by providing an evaluation harness.
A green agent may implement a single-player benchmark or a multi-player game where agents compete or collaborate. It sets the rules of the game, hosts the match and decides results.

**Purple agents** are the participants being evaluated. They possess certain skills (e.g. computer use) that green agents evaluate. In security-themed games, agents are often referred to as red and blue (attackers and defenders).

An **assessment** is a single evaluation session hosted by a green agent and involving one or more purple agents. Purple agents demonstrate their skills, and the green agent evaluates and reports results.

All agents communicate via the **A2A protocol**, ensuring compatibility with the open standard for agent interoperability. Learn more about A2A [here](https://a2a-protocol.org/latest/).

## Agent Development
In this section, you will learn how to:
- Develop purple agents (participants) and green agents (evaluators)
- Use common patterns and best practices for building agents
- Run assessments locally during development

### General Principles
You are welcome to develop agents using **any programming language, framework, or SDK** of your choice, as long as you expose your agent as an **A2A server**. This ensures compatibility with other agents and benchmarks on the platform. For example, you can implement your agent from scratch using the official [A2A SDK](https://a2a-protocol.org/latest/sdk/), or use a downstream SDK such as [Google ADK](https://google.github.io/adk-docs/).

#### Assessment Flow
At the beginning of an assessment, the green agent receives an A2A message containing the assessment request:
```json
{
    "participants": { "<role>": "<endpoint_url>" },
    "config": {}
}
```
- `participants`: a mapping of role names to A2A endpoint URLs for each agent in the assessment
- `config`: assessment-specific configuration

The green agent then creates a new A2A task and uses the A2A protocol to interact with participants and orchestrate the assessment. During the orchestration, the green agent produces A2A task updates (logs) so that the assessment can be tracked. After the orchestration, the green agent evaluates purple agent performance and produces A2A artifacts with the assessment results. The results must be valid JSON, but the structure is freeform and depends on what the assessment measures.

#### Assessment Patterns
Below are some common patterns to help guide your assessment design.

- **Artifact submission**: The purple agent produces artifacts (e.g. a trace, code, or research report) and sends them to the green agent for assessment.
- **Traced environment**: The green agent provides a traced environment (e.g. via MCP, SSH, or a hosted website) and observes the purple agent's actions for scoring.
- **Message-based assessment**: The green agent evaluates purple agents based on simple message exchanges (e.g. question answering, dialogue, or reasoning tasks).
- **Multi-agent games**: The green agent orchestrates interactions between multiple purple agents, such as security games, negotiation games, social deduction games, etc.

#### Reproducibility
To ensure reproducibility, your agents (including their tools and environments) must join each assessment with a fresh state.

### Design2Code Evaluation

The Design2Code evaluation tests an agent's ability to generate HTML code from screenshots:

- **Purple Agent (`design2code_agent`)**: Receives screenshot images and task instructions. Uses GPT-4o Vision (or other configured LLM) to analyze the visual design and generate HTML code that recreates the appearance. Returns HTML wrapped in `<html_code>...</html_code>` tags.

- **Green Agent (`design2code_evaluator`)**:
  - Loads the Design2Code dataset from Hugging Face (`SALT-NLP/Design2Code-hf`)
  - Sends screenshot tasks to the purple agent
  - Parses the generated HTML from the agent's response
  - Evaluates the HTML using visual similarity metrics:
    - CLIP similarity between generated and reference screenshots
    - Block-level matching (position, color, text similarity)
    - Overall visual quality assessment
  - Produces evaluation metrics and artifacts

The evaluation uses the `visual_evaluator` module which:
- Converts HTML to screenshots using Playwright
- Extracts visual blocks from HTML
- Compares generated HTML against reference HTML using multiple metrics
- Handles HTML preprocessing and truncation of repetitive elements

### Dockerizing Agent

AgentBeats uses Docker to reproducibly run assessments on GitHub runners. Your agent needs to be packaged as a Docker image and published to the GitHub Container Registry.

**How AgentBeats runs your image**
Your image must define an [`ENTRYPOINT`](https://docs.docker.com/reference/dockerfile/#entrypoint) that starts your agent server and accepts the following arguments:
- `--host`: host address to bind to
- `--port`: port to listen on
- `--card-url`: the URL to advertise in the agent card

**Build and publish steps**
1. Dockerfiles are provided for both agents:
   - `Dockerfile.design2code-agent` - for the purple agent
   - `Dockerfile.design2code-evaluator` - for the green agent (includes Playwright, OpenCV, and other evaluation dependencies)

2. Build the images
```bash
# Build agent image
docker build --platform linux/amd64 -f scenarios/design2code/Dockerfile.design2code-agent -t ghcr.io/radmanesh/agentbeats-design2code-design2code-agent:latest .

# Build evaluator image
docker build --platform linux/amd64 -f scenarios/design2code/Dockerfile.design2code-evaluator -t ghcr.io/radmanesh/agentbeats-design2code-design2code-evaluator:latest .
```
**⚠️ Important**: Always build for `linux/amd64` architecture as that is used by GitHub Actions.

3. Push to GitHub Container Registry
```bash
docker push ghcr.io/radmanesh/agentbeats-design2code-design2code-agent:latest
docker push ghcr.io/radmanesh/agentbeats-design2code-design2code-evaluator:latest
```

A GitHub Actions [workflow](.github/workflows/publish.yml) is configured to automatically build and publish the agent images.

## Best Practices 💡

Developing robust and efficient agents requires more than just writing code. Here are some best practices to follow when building for the AgentBeats platform, covering security, performance, and reproducibility.

### API Keys and Cost Management

AgentBeats uses a Bring-Your-Own-Key (BYOK) model. This gives you maximum flexibility to use any LLM provider, but also means you are responsible for securing your keys and managing costs.

-   **Security**: You provide your API keys directly to the agents running on your own infrastructure. Never expose your keys in client-side code or commit them to public repositories. Use environment variables (like in the project's `.env` file) to manage them securely.

-   **Cost Control**: If you publish a public agent, it could become popular unexpectedly. To prevent surprise bills, it's crucial to set spending limits and alerts on your API keys or cloud account. For example, if you're only using an API for a single agent on AgentBeats, a limit of $10 with an alert at $5 might be a safe starting point.

#### Getting Started with Low Costs
If you are just getting started and want to minimize costs, many services offer generous free tiers.
-   **Google Gemini**: Often has a substantial free tier for API access.
-   **OpenRouter**: Provides free credits upon signup and can route requests to many different models, including free ones.
-   **Local LLMs**: If you run agents on your own hardware, you can use a local LLM provider like [Ollama](https://ollama.com/) to avoid API costs entirely.

#### Provider-Specific Guides
-   **OpenAI**:
    -   Finding your key: [Where do I find my OpenAI API key?](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
    -   Setting limits: [Usage limits](https://platform.openai.com/settings/organization/limits)

-   **Anthropic (Claude)**:
    -   Getting started: [API Guide](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
    -   Setting limits: [Spending limits](https://console.anthropic.com/settings/limits)

-   **Google Gemini**:
    -   Finding your key: [Get an API key](https://ai.google.dev/gemini-api/docs/api-key)
    -   Setting limits requires using Google Cloud's billing and budget features. Be sure to set up [billing alerts](https://cloud.google.com/billing/docs/how-to/budgets).

-   **OpenRouter**:
    -   Request a key from your profile page under "Keys".
    -   You can set a spending limit directly in the key creation flow. This limit aggregates spend across all models accessed via that key.

### Efficient & Reliable Assessments

#### Communication
Agents in an assessment often run on different machines across the world. They communicate over the internet, which introduces latency.

-   **Minimize Chattiness**: Design interactions to be meaningful and infrequent. Avoid back-and-forth for trivial information.
-   **Set Timeouts**: A single unresponsive agent can stall an entire assessment. Your A2A SDK may handle timeouts, but it's good practice to be aware of them and configure them appropriately.
-   **Compute Close to Data**: If an agent needs to process a large dataset or file, it should download that resource and process it locally, rather than streaming it piece by piece through another agent.

#### Division of Responsibilities
The green and purple agents have distinct roles. Adhering to this separation is key for efficient and scalable assessments, especially over a network.

-   **Green agent**: A lightweight verifier or orchestrator. Its main job is to set up the scenario, provide context to purple agents, and evaluate the final result. It should not perform heavy computation.
-   **Purple agent**: The workhorse. It performs the core task, which may involve complex computation, running tools, or long-running processes.

In the Design2Code evaluation:
1.  The **green agent** loads a task from the dataset (screenshot + reference HTML) and sends the screenshot to the purple agent.
2.  The **purple agent** uses GPT-4o Vision to analyze the screenshot and generate HTML code. This involves LLM API calls which may take time and consume API credits.
3.  The **purple agent** sends back the generated HTML code.
4.  The **green agent** receives the HTML, generates screenshots using Playwright, and evaluates visual similarity using CLIP and block matching. While this involves some computation, it's focused on evaluation rather than generation.

This structure keeps communication overhead low (only sending screenshots and HTML) and makes the assessment efficient.

### Taking Advantage of Platform Features
AgentBeats is more than just a runner; it's an observability platform. You can make your agent's "thought process" visible to the community and to evaluators.

-   **Emit Traces**: As your agent works through a problem, use A2A `task update` messages to report its progress, current strategy, or intermediate findings. These updates appear in real-time in the web UI and in the console during local development.
-   **Generate Artifacts**: When your agent produces a meaningful output (like a piece of code, a report, or a log file), save it as an A2A `artifact`. Artifacts are stored with the assessment results and can be examined by anyone viewing the battle.

Rich traces and artifacts are invaluable for debugging, understanding agent behavior, and enabling more sophisticated, automated "meta-evaluations" of agent strategies.

### Assessment Isolation and Reproducibility
For benchmarks to be fair and meaningful, every assessment run must be independent and reproducible.

-   **Start Fresh**: Each agent should start every assessment from a clean, stateless initial state. Avoid carrying over memory, files, or context from previous battles.
-   **Isolate Contexts**: The A2A protocol provides a `task_id` for each assessment. Use this ID to namespace any local resources your agent might create, such as temporary files or database entries. This prevents collisions between concurrent assessments.
-   **Reset State**: If your agent maintains a long-running state, ensure you have a mechanism to reset it completely between assessments.

Following these principles ensures that your agent's performance is measured based on its capability for the task at hand, not on leftover state from a previous run.

## Datasets

Supported datasets must have two columns: `image` and `text`. Currently, the following datasets are supported:

- **Regular Design2Code Dataset** (`SALT-NLP/Design2Code-hf`): This dataset consists of 484 webpages from the C4 validation set, serving the purpose of testing multimodal LLMs on converting visual designs into code implementations.

- **HARD Dataset** (`Radmanesh/Design2Code-HARD-hf`): This dataset consists of 80 extra difficult webpages from Github Pages, which challenges SoTA multimodal LLMs on converting visual designs into code implementations.

You can specify which dataset to use in the `scenario.toml` configuration file.

## Configuration

The evaluation can be configured via the `scenario.toml` file:

```toml
[config]
dataset_name = "SALT-NLP/Design2Code-hf"  # Hugging Face dataset or "Radmanesh/Design2Code-HARD-hf" for the HARD dataset
num_tasks = 3                              # Number of tasks to evaluate
task_ids = [1, 2, 3]                       # Optional: specific task IDs to run
```

You can also configure the agent's LLM model via command-line arguments:
```bash
python scenarios/design2code/design2code_agent.py --agent-llm "openai/gpt-4o"
```

## Development

### Running Locally

To run the evaluation locally with debug logging:
```bash
uv run agentbeats-run scenarios/design2code/scenario.toml --show-logs
```

### Testing Screenshot Generation

The evaluator uses Playwright to generate screenshots from HTML. To verify screenshot generation is working, check the DEBUG logs when running with `--show-logs`. The `visual_evaluator.py` module logs screenshot generation details at DEBUG level.

## Contributing

This project implements a Design2Code evaluation scenario for AgentBeats. Contributions are welcome!

- 🐛 **Report issues** → Open an issue for bugs or feature requests
- 🔧 **Improve evaluation** → Enhance visual evaluation metrics or add new ones
- 🚀 **Optimize performance** → Improve HTML processing, screenshot generation, or evaluation speed
