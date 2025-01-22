from typing import Dict

class BaseStrategy:
    async def apply_async(self, source_document: str, claim: str) -> Dict[str, str]:
        raise NotImplementedError("This method should be implemented by subclasses.")
