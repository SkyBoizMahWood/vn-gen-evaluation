# vn-gen-evaluation
A evaluation algorithm for generated visual novels

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- API keys for Anthropic, Google (for Gemini), and OpenAI (optional)

## Setup

1. Clone the repository
2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then fill in the required API keys in `.env`

3. Start the Neo4j database:
   ```bash
   docker-compose up -d
   ```

## Usage

The project provides several commands for evaluating and analyzing visual novel stories:

### Run evaluation for a single story
```bash
python main.py run-evaluation-with --story-id <story_id> --trial-id <trial_id> --model-name <model_name>
```

### Batch evaluation for multiple stories
```bash
python main.py batch-run-evaluation-with --story-ids-list-path <path_to_file> --trial-id <trial_id> --model-name <model_name>
```

### Analyze evaluation results
```bash
python main.py run-analysis-with --story-id <story_id> --trial-id <trial_id>
```

## Parameters

- `story_id`: ID of the story to evaluate
- `trial_id`: Identifier for the evaluation trial
- `model_name`: Name of the generative model to use (default: "gemini-2.0-flash-001")
- `story_ids_list_path`: Path to a text file containing story IDs (one per line)
