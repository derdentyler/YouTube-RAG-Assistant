
# Video RAG API

This project is an API for handling requests using Retrieval-Augmented Generation (RAG). The API accepts video URLs and questions, generating answers using the RAG model and PostgreSQL database. The project uses FastAPI for request handling, PostgreSQL with the pgvector extension for storing vector data, and Docker for containerization.

## Installation and Setup

### Step 1: Install Poetry

To install the project dependencies, you need to use **Poetry**. If you don't have it installed yet, you can install Poetry using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Step 2: Install Project Dependencies

Once Poetry is installed, navigate to the project root directory and run the following command to install all dependencies:

```bash
poetry install
```

### Step 3: Download the Model

Before running the API, you need to download the model in **.gguf** format and save it in the `models` folder. Use the provided link to download the model and save the file in the `models/` directory of your project.

For example, load [saiga_llama3_8b-q4_k_m.gguf](https://huggingface.co/itlwas/saiga_llama3_8b-Q4_K_M-GGUF/resolve/main/saiga_llama3_8b-q4_k_m.gguf?download=true)

### Step 4: Configure the Settings

The project uses a configuration file where important parameters such as the model path, model settings, and retriever settings are specified. Update the config according to your preferences.

Example configuration (`config.yaml`):

```yaml
language: "ru"

models:
  ru:
    backend: "llama.cpp"
    model_path: "./models/saiga_llama3_8b-q4_k_m.gguf"
    n_ctx: 8192
  en:
    backend: "transformers"
    model_name: "mistralai/Mistral-7B-Instruct-v0.1"

embedding_model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

retriever:
  top_k: 5
  similarity_metric: "cosine"

# Subtitle block duration in seconds (used for merging short phrases)
subtitle_block_duration: 60
```

### Step 5: Run the Application

To start the API, run the following command:

```bash
poetry run start-api
```

This command will launch the FastAPI application using Uvicorn, and you can access the API at `http://localhost:8000`.

### Step 6: Testing the API

The API supports Swagger UI for testing all available endpoints. To access Swagger UI, open the following link in your browser:

```
http://localhost:8000/docs
```

### Unit tests

To use unit tests:

```bash
poetry run pytest
```

## Technologies

- **FastAPI** - for creating the API
- **PostgreSQL** with **pgvector** extension - for storing vector data
- **Uvicorn** - ASGI server
- **Docker** - for containerizing the project
- **Poetry** - for dependency management

## DataBase

Create remote Postgres Database. For example, using [superbase](https://supabase.com/).

## Development

For development and testing, you need to create a `.env` file based .env.example with the following configuration:

```bash
PYTHONPATH=src
LOG_FILE=logs/app.log
SUBTITLES_DIR=downloads/subtitles
SUPABASE_URL=https://*************.supabase.co
SUPABASE_KEY=****************
USER=postgres.***************
HOST=***************.pooler.supabase.com
PORT=5432
DBNAME=postgres
```

Make sure that the `.env` file is correctly configured with your Supabase instance details and model path.

## License

This project is licensed under the MIT License.

## Future Plans

In the future, I plan to add the following features:

- **CI/CD**: To improve code quality and ensure stability with CI/CD.
- **Ranking module**: To classify prompts by their usefulness using additional ranking methods.
- **Docker**: To containerize the project for easier deployment and scaling.
- **Refactoring**: Add type annotations. Add Config pydantic-class.
- **User Interface (UI)**: To provide a user-friendly interface for interacting with the API.

## Contact

For any questions or suggestions, feel free to reach out: 📧 alexander.polybinsky@gmail.com
