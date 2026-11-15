"""Lightweight version contract shared by M4 consumers."""

ASSET_VALIDATION_SCHEMA_VERSION = 3
ASSET_VALIDATION_REPORT_SCHEMA_VERSION = 3
LEGACY_CONTACT_REPORT_SCHEMA_VERSION = 2

M4_IMPLEMENTED_USES = (
    "pick_left",
    "pick_right",
    "open",
    "close",
    "click_subject",
    "press_subject",
    "switch_on",
    "switch_off",
    "rotate_to_orientation_subject",
    "adjust_orientation_subject",
    "bimanual_grab_subject",
)
M4_DISABLED_USES = ("bimanual_grab_subject",)
M4_EXECUTABLE_USES = tuple(
    use for use in M4_IMPLEMENTED_USES if use not in M4_DISABLED_USES
)
M4_ASSET_SCOPE_USES = M4_EXECUTABLE_USES + ("relation_anchor",)


__all__ = [
    "ASSET_VALIDATION_REPORT_SCHEMA_VERSION",
    "ASSET_VALIDATION_SCHEMA_VERSION",
    "LEGACY_CONTACT_REPORT_SCHEMA_VERSION",
    "M4_ASSET_SCOPE_USES",
    "M4_DISABLED_USES",
    "M4_EXECUTABLE_USES",
    "M4_IMPLEMENTED_USES",
]
