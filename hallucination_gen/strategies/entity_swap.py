from typing import Dict
from hallucination_gen.api.ollama_api import OllamaAPI
from hallucination_gen.strategies.base_strategy import BaseStrategy


class EntitySwap(BaseStrategy):
    def __init__(self):
        self.api_client = OllamaAPI()

    async def apply_async(self, source_document: str, claim: str) -> Dict[str, str]:
        prompt = self.api_client.construct_prompt(
            """Here is a source document:
            {source_document}

            And here is its summary:
            "{claim}"

            Rewrite the summary by swapping one named entity with another. Make sure the new entity is unrelated to the original one.""",
            source_document=source_document,
            claim=claim,
        )
        system_prompt = "You are an assistant that swaps named entities in summaries to create unfaithful claims."
        transformed_claim = await self.api_client.send_prompt_async(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        claim_location = (source_document.find(claim), source_document.find(claim) + len(claim))
        transformation_location = (transformed_claim.find(claim), transformed_claim.find(claim) + len(claim))

        return {
            "transformed_claim": transformed_claim,
            "transformation_type": "entity_swap",
            "claim_location": claim_location,
            "transformation_location": transformation_location,
        }
