class BaseStrategy:
    def apply(self, source_document, claim):
        raise NotImplementedError("This method should be implemented by subclasses.")
