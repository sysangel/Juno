from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path

ENTRY_DELIMITER = "\n§\n"

class MemoryStore:
    """Bounded file-backed memory modeled after Hermes MEMORY.md / USER.md."""

    def __init__(self, home: Path, memory_limit: int = 2200, user_limit: int = 1375) -> None:
        self.home = home
        self.memory_dir = home / "memories"
        self.memory_limit = memory_limit
        self.user_limit = user_limit
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self._snapshot: dict[str, str] = {"memory": "", "user": ""}

    def load(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = self._read_entries("memory")
        self.user_entries = self._read_entries("user")
        self._snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    def system_prompt_block(self) -> str:
        parts = [p for p in [self._snapshot["user"], self._snapshot["memory"]] if p]
        return "\n\n".join(parts)

    def add(self, target: str, content: str) -> str:
        entries = self._entries(target)
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        if content in entries:
            return "entry already exists"
        limit = self._limit(target)
        new_entries = entries + [content]
        new_total = len(ENTRY_DELIMITER.join(new_entries))
        if new_total > limit:
            raise ValueError(f"{target} memory would exceed {limit} chars ({new_total}/{limit}); shorten or remove entries")
        entries.append(content)
        self._write_entries(target, entries)
        return "entry added"

    def replace(self, target: str, old_text: str, content: str) -> str:
        entries = self._entries(target)
        matches = [i for i, entry in enumerate(entries) if old_text in entry]
        if not matches:
            raise ValueError(f"no {target} memory entry matched {old_text!r}")
        if len(matches) > 1:
            raise ValueError(f"multiple {target} memory entries matched {old_text!r}; be more specific")
        candidate = entries.copy()
        candidate[matches[0]] = content.strip()
        limit = self._limit(target)
        new_total = len(ENTRY_DELIMITER.join(candidate))
        if new_total > limit:
            raise ValueError(f"replacement would exceed {limit} chars ({new_total}/{limit})")
        entries[matches[0]] = content.strip()
        self._write_entries(target, entries)
        return "entry replaced"

    def remove(self, target: str, old_text: str) -> str:
        entries = self._entries(target)
        matches = [i for i, entry in enumerate(entries) if old_text in entry]
        if not matches:
            raise ValueError(f"no {target} memory entry matched {old_text!r}")
        if len(matches) > 1:
            raise ValueError(f"multiple {target} memory entries matched {old_text!r}; be more specific")
        entries.pop(matches[0])
        self._write_entries(target, entries)
        return "entry removed"

    def list_text(self, target: str | None = None) -> str:
        targets = [target] if target else ["user", "memory"]
        blocks: list[str] = []
        for t in targets:
            entries = self._entries(t)
            used = len(ENTRY_DELIMITER.join(entries)) if entries else 0
            blocks.append(f"{t.upper()} ({used}/{self._limit(t)} chars)")
            blocks.extend(f"  {i}. {entry}" for i, entry in enumerate(entries, 1))
            if not entries:
                blocks.append("  (empty)")
        return "\n".join(blocks)

    def _entries(self, target: str) -> list[str]:
        if target == "user":
            return self.user_entries
        if target == "memory":
            return self.memory_entries
        raise ValueError("target must be 'memory' or 'user'")

    def _limit(self, target: str) -> int:
        return self.user_limit if target == "user" else self.memory_limit

    def _path(self, target: str) -> Path:
        return self.memory_dir / ("USER.md" if target == "user" else "MEMORY.md")

    def _read_entries(self, target: str) -> list[str]:
        path = self._path(target)
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        return list(dict.fromkeys(entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()))

    def _write_entries(self, target: str, entries: list[str]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(target)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")
        os.replace(tmp, path)

    def _render_block(self, target: str, entries: list[str]) -> str:
        if not entries:
            return ""
        content = ENTRY_DELIMITER.join(entries)
        limit = self._limit(target)
        pct = int(min(100, len(content) / limit * 100)) if limit else 0
        label = "USER PROFILE (who the user is)" if target == "user" else "MEMORY (your personal notes)"
        sep = "═" * 46
        return f"{sep}\n{label} [{pct}% — {len(content):,}/{limit:,} chars]\n{sep}\n{content}"


class SessionStore:
    """SQLite transcript store with basic FTS5 search."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the provider turn now runs submit_user_message
        # (which writes transcript rows) on an executor worker thread. Access is
        # serialized one turn at a time (the _awaiting gate + single executor
        # call), so this is safe without extra locking. WAL is already on below.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              title TEXT,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              started_at REAL NOT NULL,
              ended_at REAL
            );
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at REAL NOT NULL,
              FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
              content, session_id UNINDEXED, role UNINDEXED, content='messages', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
              INSERT INTO messages_fts(rowid, content, session_id, role) VALUES (new.id, new.content, new.session_id, new.role);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
              INSERT INTO messages_fts(messages_fts, rowid, content, session_id, role) VALUES('delete', old.id, old.content, old.session_id, old.role);
            END;
            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
              INSERT INTO messages_fts(messages_fts, rowid, content, session_id, role) VALUES('delete', old.id, old.content, old.session_id, old.role);
              INSERT INTO messages_fts(rowid, content, session_id, role) VALUES (new.id, new.content, new.session_id, new.role);
            END;
            """
        )
        self.conn.commit()

    def start_session(self, provider: str, model: str, title: str = "") -> str:
        sid = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.conn.execute(
            "INSERT INTO sessions(id, title, provider, model, started_at) VALUES (?, ?, ?, ?, ?)",
            (sid, title, provider, model, time.time()),
        )
        self.conn.commit()
        return sid

    def append(self, session_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        self.conn.commit()

    def end_session(self, session_id: str) -> None:
        self.conn.execute("UPDATE sessions SET ended_at=? WHERE id=?", (time.time(), session_id))
        self.conn.commit()

    def update_session_model(self, session_id: str, provider: str, model: str) -> None:
        self.conn.execute("UPDATE sessions SET provider=?, model=? WHERE id=?", (provider, model, session_id))
        self.conn.commit()

    def search(self, query: str, limit: int = 5) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT m.id, m.session_id, m.role, snippet(messages_fts, 0, '[', ']', ' … ', 12) AS snippet,
                       s.started_at, s.provider, s.model
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                JOIN sessions s ON s.id = m.session_id
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )
        )

