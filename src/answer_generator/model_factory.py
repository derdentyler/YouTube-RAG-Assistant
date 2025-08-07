from abc import ABC, abstractmethod
from typing import Union
from src.utils.logger_loader import LoggerLoader
from transformers import pipeline
from llama_cpp import Llama
import gc
from src.core.abstractions.llm import BaseLLM


class TransformersLLM(BaseLLM):
    """Реализация для Hugging Face Transformers"""

    def __init__(self, model_name: str):
        self.logger = LoggerLoader.get_logger()
        try:
            # поддержка stream=True для потоковой генерации
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                return_full_text=False
            )
            self.logger.info(f"Loaded Transformers model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading Transformers model: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 200) -> str:
        try:
            output = self.pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=1
            )
            return output[0]['generated_text']
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return ""

    def stream_generate(self, prompt: str, max_length: int = 200):
        """Потоковая генерация текстовых фрагментов"""
        try:
            # Используем параметр stream=True, если поддерживается
            for out in self.pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                stream=True
            ):
                # out может быть словарём с частичным текстом
                text = out.get('generated_text') or out.get('text', '')
                yield text
        except Exception as e:
            self.logger.error(f"Stream generation error: {e}")
            return


class LlamaCppLLM(BaseLLM):
    """Реализация для GGUF моделей через llama.cpp"""

    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.logger = LoggerLoader.get_logger()
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                verbose=False
            )
            self.logger.info(f"Loaded GGUF model: {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading GGUF model: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 200) -> str:
        try:
            output = self.llm(
                prompt,
                max_tokens=max_length,
                echo=False
            )
            # Проверяем формат вывода
            if isinstance(output, dict) and "choices" in output:
                return output['choices'][0]['text'].strip()
            elif isinstance(output, str):
                return output.strip()
            else:
                return str(output)
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return ""

    def stream_generate(self, prompt: str, max_length: int = 200):
        """Потоковая генерация через llama.cpp stream=True"""
        try:
            for out in self.llm(
                prompt,
                max_tokens=max_length,
                stream=True
            ):
                # out — словарь с токеном
                if isinstance(out, dict) and 'choices' in out:
                    chunk = out['choices'][0].get('text', '')
                else:
                    chunk = str(out)
                yield chunk
        except Exception as e:
            self.logger.error(f"Stream generation error: {e}")
            return


def model_factory(config: dict) -> BaseLLM:
    """Фабрика для создания LLM"""
    lang = config.get("language", "en")
    model_config = config["models"][lang]

    if model_config["backend"] == "transformers":
        return TransformersLLM(model_config["model_name"])
    elif model_config["backend"] == "llama.cpp":
        return LlamaCppLLM(
            model_config["model_path"],
            n_ctx=model_config.get("n_ctx", 2048)
        )
    else:
        raise ValueError(f"Unknown backend: {model_config['backend']}")
