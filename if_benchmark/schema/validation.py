"""Context-aware validation for connected BenchmarkV1 documents."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import KIND_TO_PAYLOAD
from .serialization import load_document


def validate_bundle(
    documents: Iterable[Mapping[str, Any] | Path],
) -> dict[str, Any]:
    """Validate documents individually and then validate their cross-references.

    A bundle is validation context, not another persisted BenchmarkV1 IR.
    """

    from .cli import validate_document

    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, raw in enumerate(documents):
        document = load_document(raw) if isinstance(raw, Path) else dict(raw)
        normalized.append(document)
        report = validate_document(document)
        for error in report["errors"]:
            errors.append(
                {
                    "document": index,
                    "kind": document.get("kind"),
                    **error,
                }
            )

    if errors:
        return _bundle_report(normalized, errors)

    by_kind: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    by_id: dict[str, dict[str, tuple[int, dict[str, Any]]]] = defaultdict(dict)
    for index, document in enumerate(normalized):
        kind = str(document["kind"])
        payload = document[KIND_TO_PAYLOAD[kind]]
        by_kind[kind].append((index, payload))
        if isinstance(payload, dict) and payload.get("id"):
            payload_id = str(payload["id"])
            if payload_id in by_id[kind]:
                errors.append(
                    _error(
                        index,
                        kind,
                        ["id"],
                        f"duplicate {kind} id in bundle: {payload_id}",
                        "bundle_duplicate_id",
                    )
                )
            else:
                by_id[kind][payload_id] = (index, payload)

    archetypes = by_id["archetype"]
    role_graphs = by_id["role_graph"]
    families = by_id["abstract_family"]
    assignments = by_id["assignment_candidate"]
    realized_families = by_id["realized_scene_family"]

    for index, family in by_kind["abstract_family"]:
        for archetype_id in family.get("archetypes", []):
            if archetypes and archetype_id not in archetypes:
                errors.append(
                    _error(
                        index,
                        "abstract_family",
                        ["archetypes"],
                        f"unknown archetype id in bundle: {archetype_id}",
                        "bundle_archetype_ref",
                    )
                )
        role_graph_id = family.get("role_graph_id")
        if role_graph_id is not None and role_graphs and role_graph_id not in role_graphs:
            errors.append(
                _error(
                    index,
                    "abstract_family",
                    ["role_graph_id"],
                    f"unknown role graph id in bundle: {role_graph_id}",
                    "bundle_role_graph_ref",
                )
            )

    for index, assignment in by_kind["assignment_candidate"]:
        family_id = assignment.get("abstract_family_id")
        family_entry = families.get(str(family_id))
        if families and family_entry is None:
            errors.append(
                _error(
                    index,
                    "assignment_candidate",
                    ["abstract_family_id"],
                    f"unknown abstract family id in bundle: {family_id}",
                    "bundle_family_ref",
                )
            )
        expected_archetypes = set(assignment.get("archetype_ids", []))
        if archetypes:
            for archetype_id in expected_archetypes:
                if archetype_id not in archetypes:
                    errors.append(
                        _error(
                            index,
                            "assignment_candidate",
                            ["archetype_ids"],
                            f"unknown archetype id in bundle: {archetype_id}",
                            "bundle_archetype_ref",
                        )
                    )
        if family_entry is not None:
            family_archetypes = set(family_entry[1].get("archetypes", []))
            if expected_archetypes != family_archetypes:
                errors.append(
                    _error(
                        index,
                        "assignment_candidate",
                        ["archetype_ids"],
                        "assignment archetype_ids do not match abstract family",
                        "bundle_archetype_alignment",
                    )
                )
        role_ids, constraint_ids, _ = _roles_constraints_and_variants(
            expected_archetypes, archetypes, role_graphs
        )
        assigned_roles = set(assignment.get("roles", {}))
        if role_ids and assigned_roles != role_ids:
            errors.append(
                _error(
                    index,
                    "assignment_candidate",
                    ["roles"],
                    f"assignment roles {sorted(assigned_roles)} do not match "
                    f"expected roles {sorted(role_ids)}",
                    "bundle_assignment_roles",
                )
            )
        for fact_index, fact in enumerate(assignment.get("proven_facts", [])):
            constraint_id = fact.get("constraint") if isinstance(fact, dict) else None
            if constraint_ids and constraint_id not in constraint_ids:
                errors.append(
                    _error(
                        index,
                        "assignment_candidate",
                        ["proven_facts", fact_index, "constraint"],
                        f"unknown constraint id in bundle: {constraint_id}",
                        "bundle_constraint_ref",
                    )
                )

    for index, realized in by_kind["realized_scene_family"]:
        assignment_id = str(realized.get("assignment_id"))
        assignment_entry = assignments.get(assignment_id)
        if assignments and assignment_entry is None:
            errors.append(
                _error(
                    index,
                    "realized_scene_family",
                    ["assignment_id"],
                    f"unknown assignment id in bundle: {assignment_id}",
                    "bundle_assignment_ref",
                )
            )
        if assignment_entry is not None:
            assignment = assignment_entry[1]
            if realized.get("abstract_family_id") != assignment.get("abstract_family_id"):
                errors.append(
                    _error(
                        index,
                        "realized_scene_family",
                        ["abstract_family_id"],
                        "realized family and assignment abstract_family_id differ",
                        "bundle_family_alignment",
                    )
                )
            if set(realized.get("role_bindings", {})) != set(assignment.get("roles", {})):
                errors.append(
                    _error(
                        index,
                        "realized_scene_family",
                        ["role_bindings"],
                        "realized role bindings do not match assignment roles",
                        "bundle_role_alignment",
                    )
                )
            _, _, variant_schema_ids = _roles_constraints_and_variants(
                set(assignment.get("archetype_ids", [])), archetypes, role_graphs
            )
            for variant_index, variant in enumerate(realized.get("goal_variants", [])):
                schema_id = variant.get("schema_id") if isinstance(variant, dict) else None
                if variant_schema_ids and schema_id not in variant_schema_ids:
                    errors.append(
                        _error(
                            index,
                            "realized_scene_family",
                            ["goal_variants", variant_index, "schema_id"],
                            f"unknown variant schema id in bundle: {schema_id}",
                            "bundle_variant_schema_ref",
                        )
                    )

    for index, scene in by_kind["scene_spec"]:
        realized_id = str(scene.get("realized_family_id"))
        realized_entry = realized_families.get(realized_id)
        if realized_families and realized_entry is None:
            errors.append(
                _error(
                    index,
                    "scene_spec",
                    ["realized_family_id"],
                    f"unknown realized family id in bundle: {realized_id}",
                    "bundle_realized_family_ref",
                )
            )
            continue
        if realized_entry is None:
            continue
        realized = realized_entry[1]
        expected_entities = {
            template.get("key")
            for template in realized.get("templates", [])
            if isinstance(template, dict)
        }
        actual_entities = set(scene.get("entities", {}))
        if actual_entities != expected_entities:
            errors.append(
                _error(
                    index,
                    "scene_spec",
                    ["entities"],
                    f"scene entities {sorted(actual_entities)} do not match "
                    f"realized entities {sorted(expected_entities)}",
                    "bundle_scene_entities",
                )
            )
        realized_bindings = realized.get("role_bindings", {})
        for entity_key, entity in scene.get("entities", {}).items():
            if not isinstance(entity, dict):
                continue
            role_ref = entity.get("role_ref")
            binding = realized_bindings.get(role_ref)
            if not isinstance(binding, dict):
                errors.append(
                    _error(
                        index,
                        "scene_spec",
                        ["entities", entity_key, "role_ref"],
                        f"scene entity references unknown realized role: {role_ref}",
                        "bundle_scene_entity_alignment",
                    )
                )
                continue
            for field in ("entity_key", "asset", "model_id"):
                expected = entity_key if field == "entity_key" else entity.get(field)
                actual = binding.get(field)
                if actual != expected:
                    errors.append(
                        _error(
                            index,
                            "scene_spec",
                            ["entities", entity_key, field],
                            f"scene entity {field} does not match realized binding",
                            "bundle_scene_entity_alignment",
                        )
                    )
        expected_units = {
            unit.get("id")
            for unit in realized.get("units", [])
            if isinstance(unit, dict)
        }
        actual_units = {
            row.get("unit_id")
            for row in scene.get("unit_poses", [])
            if isinstance(row, dict)
        }
        if actual_units != expected_units:
            errors.append(
                _error(
                    index,
                    "scene_spec",
                    ["unit_poses"],
                    f"scene unit ids {sorted(actual_units)} do not match "
                    f"realized unit ids {sorted(expected_units)}",
                    "bundle_scene_units",
                )
            )
        expected_variants = {
            variant.get("id")
            for variant in realized.get("goal_variants", [])
            if isinstance(variant, dict)
        }
        actual_variants = set(scene.get("goal_variant_ids", []))
        from if_benchmark.layout.manipulation import valid_domain
        if not valid_domain(realized, scene):
            errors.append(
                _error(
                    index,
                    "scene_spec",
                    ["goal_variant_ids"],
                    f"scene goal variants {sorted(actual_variants)} do not match "
                    f"realized variants {sorted(expected_variants)}",
                    "bundle_scene_variants",
                )
            )

    return _bundle_report(normalized, errors)


def _roles_constraints_and_variants(
    archetype_ids: set[str],
    archetypes: dict[str, tuple[int, dict[str, Any]]],
    role_graphs: dict[str, tuple[int, dict[str, Any]]],
) -> tuple[set[str], set[str], set[str]]:
    roles: set[str] = set()
    constraints: set[str] = set()
    variants: set[str] = set()
    for archetype_id in archetype_ids:
        entry = archetypes.get(str(archetype_id))
        if entry is None:
            continue
        payload = entry[1]
        roles.update(
            str(role["id"])
            for role in payload.get("roles", [])
            if isinstance(role, dict) and role.get("id")
        )
        constraints.update(
            str(constraint["id"])
            for constraint in payload.get("constraints", [])
            if isinstance(constraint, dict) and constraint.get("id")
        )
        variants.update(
            str(variant["id"])
            for variant in payload.get("variant_schemas", [])
            if isinstance(variant, dict) and variant.get("id")
        )
    # A bundle may provide a materialized RoleGraph instead of archetype
    # definitions.  Include it when there is exactly one unambiguous graph.
    if not roles and len(role_graphs) == 1:
        payload = next(iter(role_graphs.values()))[1]
        roles.update(
            str(role["id"])
            for role in payload.get("roles", [])
            if isinstance(role, dict) and role.get("id")
        )
        constraints.update(
            str(constraint["id"])
            for constraint in payload.get("constraints", [])
            if isinstance(constraint, dict) and constraint.get("id")
        )
        variants.update(
            str(variant["id"])
            for variant in payload.get("variant_schemas", [])
            if isinstance(variant, dict) and variant.get("id")
        )
    return roles, constraints, variants


def _bundle_report(
    documents: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(str(document.get("kind")) for document in documents)
    return {
        "schema_version": 1,
        "valid": not errors,
        "document_count": len(documents),
        "document_count_by_kind": dict(sorted(counts.items())),
        "errors": errors,
    }


def _error(
    document: int,
    kind: str,
    path: list[Any],
    message: str,
    rule: str,
) -> dict[str, Any]:
    return {
        "document": document,
        "kind": kind,
        "path": "$." + ".".join(str(part) for part in path),
        "message": message,
        "rule": rule,
    }
