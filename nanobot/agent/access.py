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
        """Return *None* (all) for admins, or the filtered list for normal users."""
        if self.is_admin(sender_id):
            return None
        return list(self._data.get("user_allowed_tools", []))

    def get_allowed_skills(self, sender_id: str) -> list[str] | None:
        if self.is_admin(sender_id):
            return None
        return list(self._data.get("user_allowed_skills", []))

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
                return
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load access.json, using defaults: {}", e)
        self._data = dict(self._DEFAULT)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
