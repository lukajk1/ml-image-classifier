from collections import OrderedDict
from hashlib import sha256
import json
import torch
from transformers import AutoTokenizer, SiglipTextModel
from tag_worker import verify_bundle


class SiglipTagSuggester:
    def __init__(self, root, directory, spec, bundle):
        torch.set_num_threads(2)
        self.model = SiglipTextModel.from_pretrained(directory, local_files_only=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=False)
        self.bundle, self.spec, self.root = bundle, spec, root
        self.cache = OrderedDict()
        self.vision = None

    def embed_text(self, text):
        """Embeds a search query into the image embedding space."""
        tokens = self.tokenizer([self.bundle['prompt'].format(text)],
            padding='max_length',max_length=64,truncation=True,return_tensors='pt')
        with torch.inference_mode():
            vector = self.model(**tokens).pooler_output.float()
        return torch.nn.functional.normalize(vector,dim=-1)[0].tolist()

    def suggest(self, image_bytes, candidates, vector):
        if vector is None:
            if self.vision is None:
                from ml_adapters.siglip import SiglipEmbedder
                directory = self.root/self.spec['visionDirectory']
                bundle = json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
                key = sha256(json.dumps({'bundle':bundle,'adapter':'ml_adapters.siglip:SiglipEmbedder'},sort_keys=True).encode()).hexdigest()
                if key != self.bundle['compatibleEmbeddingKey']:
                    raise ValueError('Bundled image and text encoders do not match.')
                verify_bundle(directory,bundle)
                self.vision = SiglipEmbedder(directory)
            vector = self.vision.embed(image_bytes)
        image = torch.tensor(vector, dtype=torch.float32)
        if image.shape != (self.bundle['dimension'],) or not torch.isfinite(image).all() or image.norm() <= 0:
            raise ValueError('Invalid image embedding.')
        image = torch.nn.functional.normalize(image,dim=0)
        missing = [label for label in candidates if label not in self.cache]
        for start in range(0,len(missing),32):
            labels = missing[start:start+32]
            tokens = self.tokenizer([self.bundle['prompt'].format(label) for label in labels],
                padding='max_length',max_length=64,truncation=True,return_tensors='pt')
            with torch.inference_mode():
                vectors = torch.nn.functional.normalize(self.model(**tokens).pooler_output.float(),dim=-1)
            for label,text_vector in zip(labels,vectors):
                self.cache[label] = text_vector
        scores = [{'label':label,'score':float(torch.dot(image,self.cache[label]).clamp(-1,1))}
                  for label in candidates]
        scores.sort(key=lambda row:row['score'],reverse=True)
        # Keep the text cache bounded across libraries and vocabularies.
        while len(self.cache)>2048:
            self.cache.popitem(last=False)
        return scores[:12]
