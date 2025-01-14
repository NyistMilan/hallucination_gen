class StrategyManager:
    def __init__(self, strategies):
        self.strategies = strategies

    def apply_transformations(self, source_document, claim):
        results = []
        for strategy in self.strategies:
            result = strategy.apply(source_document, claim)
            if result:
                result.update({
                    "source_document": source_document,
                    "claim": claim,
                })
                results.append(result)
        return results
