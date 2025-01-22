import asyncio
from typing import List, Dict

class StrategyManager:
    def __init__(self, strategies):
        self.strategies = strategies

    async def apply_transformations_async(self, source_document: str, claim: str) -> List[Dict]:
        tasks = [
            strategy.apply_async(source_document, claim)
            for strategy in self.strategies
        ]
        results = await asyncio.gather(*tasks)

        enriched_results = [
            {**result, "source_document": source_document, "claim": claim}
            for result in results if result
        ]
        return enriched_results

    def apply_transformations(self, source_document: str, claim: str) -> List[Dict]:
        return asyncio.run(self.apply_transformations_async(source_document, claim))
