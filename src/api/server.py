import uvicorn
import os
from dotenv import load_dotenv


def main() -> None:
    """Запуск FastAPI через Uvicorn с параметрами из .env"""
    load_dotenv()

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("RELOAD", "True") == "True"
    )


if __name__ == "__main__":
    main()
