"""Generic suggestion protocol: adapters rank labels, never mutate tags or catalogs."""
from contextlib import redirect_stdout
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys
from runtime_paths import asset_root


def verify_bundle(directory, bundle):
    for filename, expected in bundle['files'].items():
        path = (directory/filename).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Invalid bundled tag model path.')
        digest = sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Damaged bundled model file: {filename}')


class TagWorker:
    def __init__(self, root):
        self.root = Path(root)
        self.spec = json.loads((self.root/'tagging.json').read_text(encoding='utf-8'))
        self.assets = asset_root(self.root)
        self.directory = self.assets/self.spec['directory']
        self.bundle = json.loads((self.directory/'bundle.json').read_text(encoding='utf-8'))
        self.version = sha256(json.dumps({'bundle':self.bundle, 'adapter':self.spec['adapter']},sort_keys=True).encode()).hexdigest()
        self.vocabulary = json.loads((self.root/self.spec['vocabulary']).read_text(encoding='utf-8'))
        self.adapter = None

    def dispatch(self, request):
        metadata = {'modelId':self.bundle['modelId'], 'modelVersion':self.version,
                    'compatibleEmbeddingKey':self.bundle.get('compatibleEmbeddingKey')}
        if request['method'] == 'tags.info':
            return metadata
        if self.adapter is None:
            verify_bundle(self.directory, self.bundle)
            module, factory = self.spec['adapter'].split(':',1)
            with redirect_stdout(sys.stderr):
                self.adapter = getattr(import_module(module),factory)(self.assets,self.directory,self.spec,self.bundle)
        if request['method'] == 'tags.load':
            return metadata
        if request['method'] == 'tags.embedText':
            text = request.get('text')
            if not isinstance(text, str) or not text.strip() or len(text) > 300:
                raise ValueError('Search text must be 1 to 300 characters.')
            with redirect_stdout(sys.stderr):
                vector = self.adapter.embed_text(text.strip())
            return {**metadata, 'text': text.strip(), 'vector': vector}
        if request['method'] != 'tags.suggest':
            raise ValueError('Unknown tag suggestion method.')
        path = Path(request['imagePath'])
        if not path.is_absolute() or not path.is_file() or path.stat().st_size > 100*1024*1024:
            raise ValueError('Image missing or exceeds 100 MiB.')
        image_bytes = path.read_bytes()
        if sha256(image_bytes).hexdigest() != request['contentHash']:
            raise ValueError('Image changed since import; suggestions discarded.')
        candidates, seen = [], set()
        # User vocabulary takes priority over the bundled defaults when capped.
        defaults = self.vocabulary if request.get('includeDefaults', True) else []
        for label in [*request.get('candidates',[]), *defaults]:
            if not isinstance(label,str):
                continue
            label = label.strip()
            if label and len(label)<=120 and label.casefold() not in seen:
                seen.add(label.casefold()); candidates.append(label)
            if len(candidates)>=512:
                break
        with redirect_stdout(sys.stderr):
            scores = self.adapter.suggest(image_bytes, candidates,
                request.get('vector') if request.get('embeddingKey') == metadata['compatibleEmbeddingKey'] else None)
        return {**metadata, 'assetId':request['assetId'],'contentHash':request['contentHash'], 'suggestions':scores}
