"""Role-based access control for agent users."""

import json
from pathlib import Path
from typing import Any

from loguru import logger


class AccessManager:
    """Manages admin/normal user roles and per-role tool/skill permissions.

    Data is persisted in ``workspace/access.json``.
    """

    _DEFAULT: dict[str, Any] = {
        "admin_passphrase": "",
        "admins": [],
        "user_allowed_tools": [],
        "user_allowed_skills": [],
        # Per-admin workspace-restriction overrides:
        #   True  => force restricted to workspace
        #   False => force unrestricted (full FS + shell)
        #   absent => follow the static config default
        "admin_workspace_overrides": {},
        # Per-admin self-allowlists. Absent or empty entry => no self-restriction
        # (admin gets all tools/skills). A non-empty list is enforced.
        "admin_self_allowed_tools": {},
        "admin_self_allowed_skills": {},
    }

    def __init__(self, workspace: Path):
        self._path = workspace / "access.json"
        self._data: dict[str, Any] = {}
        self._load()

    # ── Queries ───────────────────────────────────────────────────────

    def is_admin(self, sender_id: str) -> bool:
        return sender_id in self._data.get("admins", [])

    def authenticate(self, sender_id: str, passphrase: str) -> bool:
        """Validate passphrase and promote *sender_id* to admin."""
        expected = self._data.get("admin_passphrase", "")
        if not expected or passphrase != expected:
            return False
        admins: list[str] = self._data.setdefault("admins", [])
        if sender_id not in admins:
            admins.append(sender_id)
            self._save()
            logger.info("User {} authenticated as admin", sender_id)
        return True

    def get_allowed_tools(self, sender_id: str) -> list[str] | None:
        """Return *None* (all) when no allowlist applies, else the allowlist.

        Admins default to *None* (every tool enabled); an admin who has
        configured a personal self-allowlist (non-empty) gets that list applied
        to themselves only. Normal users see the shared ``user_allowed_tools``
        list, which defaults to empty — i.e. no tools until an admin grants them.
        """
        if self.is_admin(sender_id):
            self_list = self._data.get("admin_self_allowed_tools", {}).get(sender_id)
            if self_list:
                return list(self_list)
            return None
        return list(self._data.get("user_allowed_tools", []))

    def get_allowed_skills(self, sender_id: str) -> list[str] | None:
        if self.is_admin(sender_id):
            self_list = self._data.get("admin_self_allowed_skills", {}).get(sender_id)
            if self_list:
                return list(self_list)
            return None
        return list(self._data.get("user_allowed_skills", []))

    def get_workspace_override(self, sender_id: str) -> bool | None:
        """Return this admin's explicit override (True=restricted, False=unrestricted),
        or None if they follow the config default.
        """
        return self._data.get("admin_workspace_overrides", {}).get(sender_id)

    def is_workspace_restricted(self, sender_id: str | None, default: bool) -> bool:
        """Resolve effective workspace restriction for a caller.

        Normal users are always restricted. Admins follow their explicit override
        if set, otherwise the static config *default*. Unknown senders fall back
        to *default*.
        """
        if not sender_id:
            return default
        if not self.is_admin(sender_id):
            return True
        override = self.get_workspace_override(sender_id)
        if override is not None:
            return override
        return default

    def set_workspace_override(self, sender_id: str, restricted: bool | None) -> None:
        """Set or clear this admin's workspace override.

        ``restricted=True`` forces restricted-to-workspace, ``False`` forces
        unrestricted, ``None`` clears the override (revert to config default).
        """
        store: dict[str, bool] = self._data.setdefault("admin_workspace_overrides", {})
        if restricted is None:
            if sender_id in store:
                store.pop(sender_id, None)
                self._save()
            return
        if store.get(sender_id) != restricted:
            store[sender_id] = restricted
            self._save()

    # ── Per-admin self-allowlists ─────────────────────────────────────

    def get_admin_self_tools(self, sender_id: str) -> list[str] | None:
        """Return the admin's self-allowlist if configured, else None (all)."""
        lst = self._data.get("admin_self_allowed_tools", {}).get(sender_id)
        return list(lst) if lst else None

    def get_admin_self_skills(self, sender_id: str) -> list[str] | None:
        lst = self._data.get("admin_self_allowed_skills", {}).get(sender_id)
        return list(lst) if lst else None

    def toggle_admin_self_tool(
        self, sender_id: str, name: str, all_tools: list[str]
    ) -> bool:
        """Toggle *name* in this admin's self-allowlist. Returns the new state.

        On first use the list is initialized from *all_tools* (everything ON),
        then *name* is flipped off — so toggling matches the user's mental model
        of "everything available, except what I disabled".
        """
        return self._toggle_admin_self("admin_self_allowed_tools", sender_id, name, all_tools)

    def toggle_admin_self_skill(
        self, sender_id: str, name: str, all_skills: list[str]
    ) -> bool:
        return self._toggle_admin_self(
            "admin_self_allowed_skills", sender_id, name, all_skills
        )

    def _toggle_admin_self(
        self, key: str, sender_id: str, name: str, all_known: list[str]
    ) -> bool:
        store: dict[str, list[str]] = self._data.setdefault(key, {})
        lst = store.get(sender_id)
        if not lst:
            lst = [item for item in all_known if item != name]
            store[sender_id] = lst
            self._save()
            return False
        if name in lst:
            lst.remove(name)
            self._save()
            return False
        lst.append(name)
        self._save()
        return True

    def clear_admin_self(self, sender_id: str) -> None:
        """Drop this admin's tool/skill self-allowlists (back to all-allowed)."""
        changed = False
        for key in ("admin_self_allowed_tools", "admin_self_allowed_skills"):
            store: dict[str, list[str]] = self._data.get(key, {})
            if sender_id in store:
                store.pop(sender_id, None)
                changed = True
        if changed:
            self._save()

    # ── Mutations (admin-only in practice) ────────────────────────────

    def set_allowed_tools(self, tools: list[str]) -> None:
        self._data["user_allowed_tools"] = tools
        self._save()

    def toggle_tool(self, name: str) -> bool:
        """Toggle *name* in the normal-user allowed tools list. Returns new state."""
        tools: list[str] = self._data.setdefault("user_allowed_tools", [])
        if name in tools:
            tools.remove(name)
            self._save()
            return False
        tools.append(name)
        self._save()
        return True

    def set_allowed_skills(self, skills: list[str]) -> None:
        self._data["user_allowed_skills"] = skills
        self._save()

    def toggle_skill(self, name: str) -> bool:
        """Toggle *name* in the normal-user allowed skills list. Returns new state."""
        skills: list[str] = self._data.setdefault("user_allowed_skills", [])
        if name in skills:
            skills.remove(name)
            self._save()
            return False
        skills.append(name)
        self._save()
        return True

    def set_passphrase(self, new: str) -> None:
        self._data["admin_passphrase"] = new
        self._save()

    def revoke_admin(self, sender_id: str) -> bool:
        admins: list[str] = self._data.get("admins", [])
        if sender_id in admins:
            admins.remove(sender_id)
            for key in (
                "admin_workspace_overrides",
                "admin_self_allowed_tools",
                "admin_self_allowed_skills",
            ):
                self._data.get(key, {}).pop(sender_id, None)
            self._save()
            return True
        return False

    # ── Seed helper ───────────────────────────────────────────────────

    def seed_passphrase(self, passphrase: str) -> None:
        """Set passphrase only if one is not already configured."""
        if not self._data.get("admin_passphrase") and passphrase:
            self._data["admin_passphrase"] = passphrase
            self._save()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                self._migrate_legacy()
                return
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load access.json, using defaults: {}", e)
        self._data = dict(self._DEFAULT)

    def _migrate_legacy(self) -> None:
        """One-time upgrade of legacy ``unrestricted_admins`` list to overrides dict."""
        legacy = self._data.pop("unrestricted_admins", None)
        if not legacy:
            return
        store: dict[str, bool] = self._data.setdefault("admin_workspace_overrides", {})
        for sid in legacy:
            store.setdefault(sid, False)  # legacy entry meant "force unrestricted"
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
