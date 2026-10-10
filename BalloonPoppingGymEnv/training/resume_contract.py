"""Read-only provenance checks for continuing the exact mixed-scenario pilot."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_parent(checkpoint, expected_sha256, root, schema, scenarios, weights,
                    n_steps, batch_size):
    checkpoint, root = Path(checkpoint).resolve(), Path(root).resolve()
    if sha256(checkpoint) != expected_sha256:
        raise ValueError('Parent checkpoint hash mismatch')
    manifest_path = checkpoint.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('feature_schema') != schema:
        raise ValueError('Parent feature schema mismatch')
    mixture = manifest['mixture']
    if (list(scenarios) != mixture['scenarios'] or len(weights) != len(mixture['probabilities'])
            or any(abs(float(a)-float(b)) > 1e-12 for a, b in zip(weights, mixture['probabilities']))):
        raise ValueError('Continuation must preserve the parent mixture')
    if any(manifest['settings'][k] != v for k, v in [('n_steps', n_steps), ('batch_size', batch_size)]):
        raise ValueError('Continuation must preserve rollout and batch sizes')
    sources = manifest['source_sha256']
    if not sources:
        raise ValueError('Parent has no source provenance')
    for name, expected in sources.items():
        if sha256(root/name) != expected:
            raise ValueError(f'Parent source differs: {name}')
    return manifest, dict(checkpoint=str(checkpoint), checkpoint_sha256=expected_sha256,
                         manifest_sha256=sha256(manifest_path))
