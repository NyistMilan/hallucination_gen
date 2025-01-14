import requests
import json
import yaml


class OllamaAPI:
    def __init__(self, config_path="hallucinationGen/configs/ollama_api.yml"):
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.api_url = config["ollama_api"]["api_url"]
        self.model_name = config["ollama_api"]["model_name"]
        self.default_params = {
            "temperature": config["ollama_api"].get("default_temperature", 0.7),
            "max_tokens": config["ollama_api"].get("default_max_tokens", 100),
            "top_p": config["ollama_api"].get("default_top_p", 1.0),
            "frequency_penalty": config["ollama_api"].get("default_frequency_penalty", 0.0),
        }

    def send_prompt(self, prompt, system_prompt=None, **kwargs):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.default_params["temperature"],
            "max_tokens": self.default_params["max_tokens"],
            "top_p": self.default_params["top_p"],
            "frequency_penalty": self.default_params["frequency_penalty"],
        }
        if system_prompt:
            payload["system"] = system_prompt
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            choices = result.get("choices", [])
            return choices[0].get("text", "No response received.") if choices else "No choices in the API response."
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
        except json.JSONDecodeError:
            return "Error: Unable to parse the response."

    @staticmethod
    def construct_prompt(template, **placeholders):
        return template.format(**placeholders)
