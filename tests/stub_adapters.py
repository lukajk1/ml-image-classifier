"""Tiny stand-ins for the torch adapters, following the same constructor contracts."""


class StubClassifier:
    instances = 0

    def __init__(self, checkpoint, config):
        StubClassifier.instances += 1
        print("adapter chatter that must not reach the protocol stream")
        self.checkpoint = checkpoint
        self.labels = config.get("labels", ["only"])

    def predict(self, image_bytes):
        print("prediction chatter")
        return [{"label": label, "confidence": 1.0 / len(self.labels)} for label in self.labels]


class StubEmbedder:
    dimension = 4

    def __init__(self, directory):
        self.directory = directory

    def embed(self, image_bytes):
        print("embedding chatter")
        return [0.5] * StubEmbedder.dimension


class StubTagSuggester:
    last_call = None

    def __init__(self, root, directory, spec, bundle):
        self.bundle = bundle

    def embed_text(self, text):
        StubTagSuggester.last_call = {"text": text}
        return [0.5, 0.5, 0.5, 0.5]

    def suggest(self, image_bytes, candidates, vector):
        StubTagSuggester.last_call = {"candidates": candidates, "vector": vector}
        return [{"label": label, "score": 0.1} for label in candidates[:12]]
