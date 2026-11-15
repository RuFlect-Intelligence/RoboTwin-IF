"""Exact-byte historical resolution for a frozen release's mutable index.

Only run_manifest.yml may be redirected. It is not read by graph reconstruction;
all scene/task/attempt and semantic control sources retain their original paths.
"""
from pathlib import Path

from .io import InputLedger, IntegrityError, source_file_error


def _require(condition, message):
    if not condition:
        raise IntegrityError(message)


def resolve_sources(root, inputs, *, binding_path=None, live_sources=False):
    """Return effective source pins and a ledger guarding the binding itself."""
    root = Path(root)
    _require(not (binding_path is not None and live_sources),
             "source binding and live-source mode are mutually exclusive")
    revision = (root / 'benchmark_manifest.yml').resolve().parent
    frozen = revision.parent.name == '.frozen' and not revision.name.startswith('.')
    owner = revision.parent.parent if frozen else root
    default = owner / 'source_snapshot_binding.json'
    if binding_path is None and not live_sources and frozen and (default.exists() or default.is_symlink()):
        binding_path = default
    ledger = InputLedger()
    evidence = {'mode': 'live', 'binding_path': None, 'resolved_inputs': []}
    if binding_path is None:
        return list(inputs['files']), ledger, evidence
    _require(frozen, "historical source binding requires a frozen revision")
    binding_path = Path(binding_path)
    binding = ledger.load(binding_path)
    receipt = ledger.load(owner / 'M16_FROZEN.json')
    report = ledger.load(revision / 'publish_block_report.yml')
    _require(binding.get('artifact_kind') == 'm16_historical_source_snapshot_binding'
             and binding.get('version') == 1
             and binding.get('policy') == 'explicit_exact_byte_source_snapshot_for_historical_audit',
             "unsupported historical source binding")
    _require(receipt.get('publication_ready') is True and report.get('publication_ready') is True
             and Path(receipt['revision']).resolve() == revision
             and receipt['publish_block_report_hash'] == report['artifact_hash'],
             "source binding freeze receipt mismatch")
    _require(binding.get('freeze_receipt_hash') == receipt['artifact_hash']
             and binding.get('publish_block_report_hash') == report['artifact_hash']
             and binding.get('input_evidence_index_hash') == inputs['artifact_hash']
             and report['artifact_hashes']['input_evidence_index.yml'] == inputs['artifact_hash'],
             "source binding belongs to another freeze or input index")
    entries = binding.get('entries')
    _require(isinstance(entries, list) and len(entries) == 1,
             "historical index binding requires exactly one explicit entry")
    entry = entries[0]
    pin = entry['original_input']
    original = Path(pin['path'])
    _require(original.is_absolute() and original.name == 'run_manifest.yml',
             "historical binding may only resolve run_manifest.yml")
    matches = [p for p in inputs['files'] if p['path'] == pin['path']]
    _require(len(matches) == 1 and matches[0] == pin,
             "historical binding must preserve the complete original input pin")
    snapshot = Path(entry['snapshot_path'])
    if not snapshot.is_absolute():
        snapshot = binding_path.parent / snapshot
    _require(snapshot.resolve() != original.resolve(), "historical snapshot aliases the live source")
    _require(entry.get('snapshot_sha256') == pin['sha256'], "snapshot digest differs from original pin")
    effective = {**pin, 'path': str(snapshot.absolute())}
    error = source_file_error(effective)
    _require(error is None, f"invalid historical snapshot: {error}")
    ledger.pin(snapshot, pin['sha256'])
    evidence = {'mode': 'historical_snapshot', 'binding_path': str(binding_path.absolute()),
                'binding_hash': binding['artifact_hash'],
                'resolved_inputs': [{'original_path': pin['path'], 'snapshot_path': effective['path'],
                                     'sha256': pin['sha256'], 'size_bytes': pin['size_bytes']}]}
    return [effective if p['path'] == pin['path'] else p for p in inputs['files']], ledger, evidence
