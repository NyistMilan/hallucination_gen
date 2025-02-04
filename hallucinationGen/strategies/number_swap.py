from hallucinationGen.api.ollama_api import OllamaAPI
from hallucinationGen.strategies.base_strategy import BaseStrategy


class NumberSwap(BaseStrategy):
    def __init__(self):
        self.api_client = OllamaAPI()

    def apply(self, source_document, claim):
        prompt = self.api_client.construct_prompt(
            """Here is a source document:
            {source_document}

            And here is its summary:
            "{claim}"

            Rewrite the summary by replacing one number with another random number. Ensure the change is obvious but still grammatically correct.""",
            source_document=source_document,
            claim=claim,
        )
        system_prompt = "You are an assistant that swaps numbers in summaries to create unfaithful claims."
        transformed_claim = self.api_client.send_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        claim_location = (source_document.find(claim), source_document.find(claim) + len(claim))
        transformation_location = (transformed_claim.find(claim), transformed_claim.find(claim) + len(claim))

        return {
            "transformed_claim": transformed_claim,
            "transformation_type": "number_swap",
            "claim_location": claim_location,
            "transformation_location": transformation_location,
        }
