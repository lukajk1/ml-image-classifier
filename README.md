# Umbra Tags classifier

Flutter runs `runner.py` as a persistent local subprocess. It loads one model at a
time, keeps it warm between images, and shuts down with the app. There is no web
server, cloud call, image copying or file moving in this inference path.

## Use from Flutter

Right-click an image or selection and choose **ML classify** (also in Edit).
The default is `best_model.pth`. All class scores are stored in the library's
`predictions` table, with the model ID, checkpoint SHA-256, image SHA-256 and time.
The best label is added as a tag when its confidence is at least 80%. Existing tags
are never removed, including labels from earlier classification runs. Below the
threshold the result is saved without assigning a tag. A results dialog reports
each prediction, confidence and any failures. Batch cancellation stops after the
current image. Missing/changed image files are reported individually.

**Edit → Options → Machine learning** selects the model, ML folder, Python executable, confidence
threshold, or prediction-only mode. Settings persist in Flutter's session JSON.
The app searches its working directory and executable ancestors for the sibling
`umbra-tags-ml` folder; an explicit folder or `UMBRA_ML_HOME` overrides discovery.
`UMBRA_ML_PYTHON` overrides the default `.venv` executable. A packaged application
installed elsewhere needs its ML folder selected in settings. macOS inference uses
the same protocol, but launching Python from a sandboxed distributed app needs
separate packaging/entitlement work; this integration is currently built on Windows.

## Setup

A project-local CPU environment has been configured on this machine. To recreate:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

For macOS/Linux development, create `.venv` with Python 3.12 and install
`requirements.txt` into `.venv/bin/python` from the normal PyPI index. No pretrained
weights are downloaded at inference time: the adapter constructs MobileNet with
`weights=None`, then loads the complete local state dict using `weights_only=True`.

## Existing checkpoints

Both checkpoints have class order `artwork`, `photos` and use MobileNet V3 Small.

- `best_model.pth`: newer filesystem timestamp (September 16, 2026); checkpoint
  metadata records fine-tune epoch 6 and validation balanced accuracy
  **0.8452380952**. Its stored preprocessing is 224×224, ImageNet normalization,
  with EXIF transposition.
- `model.pth`: older filesystem timestamp (April 29, 2026); contains weights and
  class names but no evaluation metric. Uses the existing inference script's
  224×224/ImageNet preprocessing without EXIF transposition.

The default reflects the newer saved best checkpoint; no new comparative
benchmark was run. Confidence is a softmax score, not a calibrated accuracy claim.
The original `inference.py` and training scripts remain available and unchanged.

## Swapping models and runtimes

`models.json` owns the registry and default choice. Each entry has a stable `id`,
display `name`, `checkpoint`, and `adapter` in `module:ClassName` form. The registry
is trusted local configuration, so an adapter is ordinary executable Python code.
To add another architecture, implement:

```python
class MyClassifier:
    def __init__(self, checkpoint, config):
        self.labels = ["label-a", "label-b"]
        # Load model and its matching preprocessing here.

    def predict(self, image_bytes):
        return [{"label": "label-a", "confidence": 0.9},
                {"label": "label-b", "confidence": 0.1}]
```

The adapter receives the original encoded bytes and returns scores; it has no
library database dependency. Restart the app after editing registry/adapter code.
Changing checkpoint contents changes the recorded model version automatically.
Currently the UI policy selects the highest-confidence label. A future multi-label
policy can be added without changing how inference or prediction storage works.

For a different runtime altogether, implement Flutter's `ImageClassifier`
interface in `lib/ml/classifier.dart` and replace the `_imageClassifier` factory.
The gallery and library APIs do not depend on PyTorch.

## Code-driven library tags

These Dart APIs run transactions in the library worker isolate:

```dart
// Creates missing names, reuses existing names (SQLite NOCASE), preserves other tags.
await library.tagAssetsByName([assetId], ['artwork', 'Reference']);

// Use IDs when hierarchy or explicit tag editing matters.
final tagId = await library.saveTag(name: 'Architecture', parentId: parentId);
await library.editTags([assetId], add: [tagId]);

// Classifier-independent persistence/policy: all scores + optional best-label tag.
await library.applyClassification(
  assetId: assetId, contentHash: imageSha256,
  modelId: 'my-model', modelVersion: checkpointSha256,
  scores: [{'label': 'artwork', 'confidence': 0.95}],
  threshold: 0.8, assignTags: true,
);
final predictions = await library.predictions(assetId);
```

Assignment does not duplicate tags. A classification result with a stale image
hash or missing asset is rejected. Tags and prediction rows commit together; manual
tag definitions and existing assignments are preserved. Repeating a model/version
run replaces that version's scores; scores from other versions remain available.

## Worker protocol

One JSON request/reply per line. Requests contain `id` and `method`:

- `models`: returns `defaultModel` and `{id,name,available}` entries.
- `load`, with optional `modelId`: loads the checkpoint; returns identity and labels.
- `classify`, with `modelId`, `assetId`, absolute `imagePath`, `contentHash`: returns
  `assetId`, `contentHash`, `modelId`, `modelVersion`, `labels`, and numeric `scores`.

Replies are `{id,ok:true,result:...}` or `{id,ok:false,error:"..."}`. Diagnostics go
to stderr. Original image bytes are hashed before decoding to detect changed files.
GIF/animated inputs use their first frame. CPU inference keeps dependencies small;
the adapter is the place to add GPU/device selection later.


## Bundled similarity model

`embedding.json` selects a separate embedding adapter and local model folder.
`prepare_similarity_model.py` is a **developer/build-time** step: it downloads
`google/siglip2-base-patch16-224` revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` and exports only its vision tower
(~354 MiB), processor, attribution, Apache-2.0 license, and integrity manifest into
`models/siglip2-base-224`. Run it using `.venv/Scripts/python.exe` when preparing
another development/build machine. Ship the resulting folder with the app.

Runtime uses `local_files_only=True`, checks hashes before loading, and Flutter
sets Hugging Face/Transformers offline mode. No image data leaves the machine.
Model resources are shared across libraries; embeddings live in each catalog.
The recipe applies EXIF orientation, RGB conversion and the model's 224px processor
then saves 768-dimensional normalized pooler features. It handles the first frame
of animations. No training or tags are required.

Protocol methods: `embedding.info`, `embedding.load`, `embedding.compute`.
Compute takes assetId/imagePath/contentHash and returns vector plus image/model
identity. The Flutter library writer rejects stale/deleted images. The classifier
protocol remains unchanged and uses a separate worker process from embeddings.
To change weights/preprocessing, export a new bundle manifest; its derived key
creates an independent index. Never mix vectors across keys.

For Windows distribution use `../umbra-tags-flutter/tool/package_windows.ps1`.
It includes a portable Python runtime rather than copying virtualenv launcher
executables, which would still depend on the development machine's Python path.


## Swappable tag-suggestion adapters

Run `.venv/Scripts/python.exe prepare_tag_model.py` during development/build setup
(after `prepare_similarity_model.py`). This exports the matching pinned SigLIP
text encoder and tokenizer to `models/siglip2-text-224`, with integrity hashes,
license and attribution. The app never runs these preparation scripts. The Windows
packager includes both model folders and all required Python dependencies.

`tagging.json` selects the adapter factory, bundled model directory and vocabulary.
`tag_worker.py` implements the `tags.info`, `tags.load`, and `tags.suggest` protocol.
Adapters implement `suggest(image_bytes, candidates, compatible_vector)` and return
`[{label,score},...]`; they must not write to catalogs. An alternative model may
ignore the candidate list and generate labels. Flutter's separate `TagSuggester`
interface can also be replaced with an ONNX or other runtime without UI changes.

The default SigLIP adapter normalizes the matching text pooler features, caches
candidate vectors in memory, and ranks dot products against the normalized image
vector. It uses the model's recommended padded 64-token text inputs. Existing tag
names precede the bundled subject vocabulary; duplicate labels are folded and at
most 512 candidate labels are evaluated per image. Results are the top 12 ranked
labels, not calibrated confidence percentages and not automatic assignments.

Image vectors are reused only when their model key matches the text bundle's
`compatibleEmbeddingKey`; otherwise the tagger loads its own pinned vision tower.
The source image SHA-256 is verified even when a cached vector is supplied. The
vision bundle/embedding key is unchanged by adding text support, so existing
similarity indexes do not need rebuilding. Runtime remains entirely offline.
