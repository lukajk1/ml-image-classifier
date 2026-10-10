"""Build-time export of matching SigLIP text weights/tokenizer; never run by the app."""
import json
from hashlib import sha256
from pathlib import Path
from transformers import AutoTokenizer, SiglipTextModel
from prepare_similarity_model import ROOT, REPOSITORY, REVISION

if __name__ == '__main__':
    destination = ROOT / 'models' / 'siglip2-text-224'
    destination.mkdir(parents=True, exist_ok=True)
    model = SiglipTextModel.from_pretrained(REPOSITORY, revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(REPOSITORY, revision=REVISION, use_fast=False)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    vision = ROOT / 'models' / 'siglip2-base-224'
    bundle = json.loads((vision/'bundle.json').read_text(encoding='utf-8'))
    if bundle['revision'] != REVISION:
        raise ValueError('Vision and text model revisions must match.')
    vision_key = sha256(json.dumps({'bundle':bundle, 'adapter':'ml_adapters.siglip:SiglipEmbedder'}, sort_keys=True).encode()).hexdigest()
    for name in ['MODEL_CARD.md', 'LICENSE.txt']:
        (destination/name).write_bytes((vision/name).read_bytes())
    hashes = {}
    for path in destination.iterdir():
        if path.name in ('bundle.json','MODEL_CARD.md','LICENSE.txt') or not path.is_file():
            continue
        digest = sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    (destination/'bundle.json').write_text(json.dumps({
        'modelId':REPOSITORY, 'revision':REVISION,
        'compatibleEmbeddingKey':vision_key, 'dimension':768,
        'prompt':'This is a photo of {}.', 'recipe':'siglip-tag-ranking-v1', 'files':hashes,
    }, indent=2), encoding='utf-8')
    print(f'Bundled tag model at {destination}', flush=True)
