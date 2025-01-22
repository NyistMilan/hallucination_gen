import yaml
from typing import Optional, Dict, Any, List, Union
from ollama import AsyncClient, ChatResponse


class OllamaAPI:
    def __init__(self, config_path: str = "hallucination_gen/configs/ollama_api.yml"):
        self.config = self._load_config(config_path)
        self.model_name = self.config.get("model_name", "default")
        self.default_params = {
            "temperature": self.config.get("default_temperature", 0.7),
            "max_tokens": self.config.get("default_max_tokens", 100),
            "top_p": self.config.get("default_top_p", 1.0),
            "frequency_penalty": self.config.get("default_frequency_penalty", 0.0),
        }
        self.client = AsyncClient()

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as config_file:
                return yaml.safe_load(config_file)["ollama_api"]
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def _validate_params(self, params: Dict[str, Any]) -> None:
        if not (0.0 <= params["temperature"] <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        if not (1 <= params["max_tokens"] <= 2048):
            raise ValueError("Max tokens must be between 1 and 2048.")

    def _construct_messages(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def send_prompt_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Union[float, int, str]
    ) -> str:
        params = {**self.default_params, **kwargs}
        self._validate_params(params)

        messages = self._construct_messages(prompt, system_prompt)

        try:
            response: ChatResponse = await self.client.chat(model=self.model_name, messages=messages, **params)
            return response.message.content
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def construct_prompt(template: str, **placeholders: Any) -> str:
        try:
            return template.format(**placeholders)
        except KeyError as e:
            raise ValueError(f"Missing placeholder in template: {e}")
