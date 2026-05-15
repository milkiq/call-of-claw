from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coc.content.packages import ContentPackage, Visibility
from coc.content.registry import ContentRegistry
from coc.content.visibility import AccessMode, can_load_reference


@dataclass(frozen=True)
class RetrievedSpan:
    package_id: str
    reference_id: str
    title: str
    path: str
    text: str
    visibility: str
    score: int = 0

    def to_dict(self) -> dict[str, str | int]:
        return {
            "package_id": self.package_id,
            "reference_id": self.reference_id,
            "title": self.title,
            "path": self.path,
            "text": self.text,
            "visibility": self.visibility,
            "score": self.score,
        }


@dataclass(frozen=True)
class RetrievalResult:
    spans: list[RetrievedSpan]
    diagnostics: dict[str, Any]


def query_terms(query: str) -> list[str]:
    """Build lightweight multilingual search terms without assuming a ruleset."""

    lower = query.lower()
    terms = re.findall(r"[a-z0-9_]+", lower)
    for block in re.findall(r"[\u4e00-\u9fff]+", lower):
        if len(block) <= 2:
            terms.append(block)
        else:
            terms.extend(block[index : index + 2] for index in range(len(block) - 1))
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def _span_sort_key(span: RetrievedSpan) -> tuple[int, str, str]:
    return (-span.score, span.package_id, span.reference_id)


def search_package_text(
    package: ContentPackage,
    query: str,
    *,
    mode: AccessMode = AccessMode.GM,
    limit: int = 5,
) -> list[RetrievedSpan]:
    result = search_package_text_with_diagnostics(
        package,
        query,
        mode=mode,
        limit=limit,
    )
    return result.spans


def search_package_text_with_diagnostics(
    package: ContentPackage,
    query: str,
    *,
    mode: AccessMode = AccessMode.GM,
    limit: int = 5,
) -> RetrievalResult:
    terms = query_terms(query)
    hits: list[RetrievedSpan] = []
    files_scanned = 0
    chars_scanned = 0
    for reference in package.manifest.references:
        if not can_load_reference(reference, mode):
            continue
        path = package.reference_path(reference)
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        files_scanned += 1
        chars_scanned += len(text)
        haystack = f"{reference.title}\n{' '.join(reference.tags)}\n{text}".lower()
        score = sum(1 for term in terms if term in haystack) if terms else 1
        if score:
            hits.append(
                RetrievedSpan(
                    package_id=package.id,
                    reference_id=reference.id,
                    title=reference.title,
                    path=str(path),
                    text=text[:4000],
                    visibility=reference.visibility.value,
                    score=score,
                )
            )
    spans = sorted(hits, key=_span_sort_key)[:limit]
    return RetrievalResult(
        spans=spans,
        diagnostics={
            "search_backend": "scan",
            "files_scanned": files_scanned,
            "chars_scanned": chars_scanned,
            "retrieved_chars": sum(len(span.text) for span in spans),
            "index_rebuilt": False,
            "fallback": False,
        },
    )


def search_registry_text(
    registry: ContentRegistry,
    query: str,
    *,
    package_ids: list[str] | None = None,
    mode: AccessMode = AccessMode.GM,
    limit: int = 8,
) -> list[RetrievedSpan]:
    return search_registry_text_with_diagnostics(
        registry,
        query,
        package_ids=package_ids,
        mode=mode,
        limit=limit,
    ).spans


def search_registry_text_with_diagnostics(
    registry: ContentRegistry,
    query: str,
    *,
    package_ids: list[str] | None = None,
    mode: AccessMode = AccessMode.GM,
    limit: int = 8,
) -> RetrievalResult:
    selected_ids = set(package_ids or [])
    packages = [
        package
        for package in registry.packages
        if not selected_ids or package.id in selected_ids
    ]
    hits: list[RetrievedSpan] = []
    diagnostics = {
        "search_backend": "scan",
        "files_scanned": 0,
        "chars_scanned": 0,
        "retrieved_chars": 0,
        "index_rebuilt": False,
        "fallback": False,
    }
    for package in packages:
        result = search_package_text_with_diagnostics(package, query, mode=mode, limit=limit)
        hits.extend(result.spans)
        diagnostics["files_scanned"] += int(result.diagnostics.get("files_scanned", 0))
        diagnostics["chars_scanned"] += int(result.diagnostics.get("chars_scanned", 0))
    spans = sorted(hits, key=_span_sort_key)[:limit]
    diagnostics["retrieved_chars"] = sum(len(span.text) for span in spans)
    return RetrievalResult(spans=spans, diagnostics=diagnostics)


def search_registry_text_indexed(
    registry: ContentRegistry,
    query: str,
    *,
    sqlite_path: Path,
    package_ids: list[str] | None = None,
    mode: AccessMode = AccessMode.GM,
    limit: int = 8,
) -> RetrievalResult:
    try:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            index_rebuilt = _ensure_content_index(conn, registry)
            spans, diagnostics = _search_indexed_rows(
                conn,
                query,
                package_ids=package_ids or [],
                mode=mode,
                limit=limit,
            )
            diagnostics["index_rebuilt"] = index_rebuilt
            return RetrievalResult(spans=spans, diagnostics=diagnostics)
    except Exception as error:
        fallback = search_registry_text_with_diagnostics(
            registry,
            query,
            package_ids=package_ids,
            mode=mode,
            limit=limit,
        )
        diagnostics = {
            **fallback.diagnostics,
            "search_backend": "scan_fallback",
            "fallback": True,
            "error": str(error),
        }
        return RetrievalResult(spans=fallback.spans, diagnostics=diagnostics)


def warm_content_index(
    registry: ContentRegistry,
    *,
    sqlite_path: Path,
) -> dict[str, Any]:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        index_rebuilt = _ensure_content_index(conn, registry)
        row = conn.execute("SELECT COUNT(*) AS count FROM content_reference_index").fetchone()
    return {
        "index_rebuilt": index_rebuilt,
        "indexed_references": int(row["count"] if row else 0),
    }


def _ensure_content_index(conn: sqlite3.Connection, registry: ContentRegistry) -> bool:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS content_reference_index (
          package_id TEXT NOT NULL,
          reference_id TEXT NOT NULL,
          title TEXT NOT NULL,
          path TEXT NOT NULL,
          tags_json TEXT NOT NULL,
          visibility TEXT NOT NULL,
          package_version TEXT NOT NULL,
          manifest_mtime_ns INTEGER NOT NULL,
          reference_mtime_ns INTEGER NOT NULL,
          content_hash TEXT NOT NULL,
          text TEXT NOT NULL,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (package_id, reference_id)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS content_reference_fts USING fts5(
          package_id UNINDEXED,
          reference_id UNINDEXED,
          title,
          tags,
          text,
          visibility UNINDEXED
        );
        """
    )
    rebuilt = False
    for package in registry.packages:
        manifest_mtime_ns = package.manifest_path.stat().st_mtime_ns
        for reference in package.manifest.references:
            path = package.reference_path(reference)
            if not path.exists() or not path.is_file():
                continue
            reference_mtime_ns = path.stat().st_mtime_ns
            existing = conn.execute(
                """
                SELECT package_version, manifest_mtime_ns, reference_mtime_ns
                FROM content_reference_index
                WHERE package_id = ? AND reference_id = ?
                """,
                (package.id, reference.id),
            ).fetchone()
            fts_existing = conn.execute(
                """
                SELECT 1
                FROM content_reference_fts
                WHERE package_id = ? AND reference_id = ?
                LIMIT 1
                """,
                (package.id, reference.id),
            ).fetchone()
            if (
                existing
                and fts_existing
                and str(existing["package_version"]) == package.manifest.version
                and int(existing["manifest_mtime_ns"]) == manifest_mtime_ns
                and int(existing["reference_mtime_ns"]) == reference_mtime_ns
            ):
                continue
            text = path.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            tags_json = json.dumps(reference.tags, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO content_reference_index (
                  package_id, reference_id, title, path, tags_json, visibility,
                  package_version, manifest_mtime_ns, reference_mtime_ns,
                  content_hash, text, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(package_id, reference_id) DO UPDATE SET
                  title = excluded.title,
                  path = excluded.path,
                  tags_json = excluded.tags_json,
                  visibility = excluded.visibility,
                  package_version = excluded.package_version,
                  manifest_mtime_ns = excluded.manifest_mtime_ns,
                  reference_mtime_ns = excluded.reference_mtime_ns,
                  content_hash = excluded.content_hash,
                  text = excluded.text,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (
                    package.id,
                    reference.id,
                    reference.title,
                    str(path),
                    tags_json,
                    reference.visibility.value,
                    package.manifest.version,
                    manifest_mtime_ns,
                    reference_mtime_ns,
                    content_hash,
                    text,
                ),
            )
            conn.execute(
                """
                DELETE FROM content_reference_fts
                WHERE package_id = ? AND reference_id = ?
                """,
                (package.id, reference.id),
            )
            conn.execute(
                """
                INSERT INTO content_reference_fts
                  (package_id, reference_id, title, tags, text, visibility)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    package.id,
                    reference.id,
                    reference.title,
                    " ".join(reference.tags),
                    text,
                    reference.visibility.value,
                ),
            )
            rebuilt = True
    return rebuilt


def _search_indexed_rows(
    conn: sqlite3.Connection,
    query: str,
    *,
    package_ids: list[str],
    mode: AccessMode,
    limit: int,
) -> tuple[list[RetrievedSpan], dict[str, Any]]:
    selected_ids = set(package_ids)
    visible_values = _visible_values(mode)
    terms = query_terms(query)
    rows = _indexed_candidate_rows(conn, terms=terms, package_ids=package_ids, mode=mode)
    hits: list[RetrievedSpan] = []
    chars_scanned = 0
    for row in rows:
        if selected_ids and str(row["package_id"]) not in selected_ids:
            continue
        if str(row["visibility"]) not in visible_values:
            continue
        text = str(row["text"])
        chars_scanned += len(text)
        tags = " ".join(json.loads(row["tags_json"]))
        haystack = f"{row['title']}\n{tags}\n{text}".lower()
        score = sum(1 for term in terms if term in haystack) if terms else 1
        if score:
            hits.append(
                RetrievedSpan(
                    package_id=str(row["package_id"]),
                    reference_id=str(row["reference_id"]),
                    title=str(row["title"]),
                    path=str(row["path"]),
                    text=text[:4000],
                    visibility=str(row["visibility"]),
                    score=score,
                )
            )
    spans = sorted(hits, key=_span_sort_key)[:limit]
    return spans, {
        "search_backend": "sqlite_fts",
        "files_scanned": 0,
        "chars_scanned": chars_scanned,
        "retrieved_chars": sum(len(span.text) for span in spans),
        "fallback": False,
    }


def _indexed_candidate_rows(
    conn: sqlite3.Connection,
    *,
    terms: list[str],
    package_ids: list[str],
    mode: AccessMode,
) -> list[sqlite3.Row]:
    visible_values = _visible_values(mode)
    ascii_terms = [term for term in terms if re.fullmatch(r"[a-z0-9_]+", term)]
    if ascii_terms:
        match_query = " OR ".join(f'"{term}"' for term in ascii_terms[:8])
        params: list[Any] = [match_query]
        package_clause = _in_clause("i.package_id", package_ids, params)
        visibility_clause = _in_clause("i.visibility", list(visible_values), params)
        rows = conn.execute(
            f"""
            SELECT DISTINCT i.*
            FROM content_reference_index i
            JOIN (
              SELECT package_id, reference_id
              FROM content_reference_fts
              WHERE content_reference_fts MATCH ?
            ) f ON f.package_id = i.package_id
               AND f.reference_id = i.reference_id
            WHERE 1 = 1
            {package_clause}
            {visibility_clause}
            """,
            params,
        ).fetchall()
        if rows:
            return list(rows)
    params = []
    package_clause = _in_clause("package_id", package_ids, params)
    visibility_clause = _in_clause("visibility", list(visible_values), params)
    return list(
        conn.execute(
            f"""
            SELECT *
            FROM content_reference_index
            WHERE 1 = 1
            {package_clause}
            {visibility_clause}
            ORDER BY package_id, reference_id
            """,
            params,
        ).fetchall()
    )


def _in_clause(column: str, values: list[str], params: list[Any]) -> str:
    if not values:
        return ""
    params.extend(values)
    placeholders = ", ".join("?" for _ in values)
    return f" AND {column} IN ({placeholders})"


def _visible_values(mode: AccessMode) -> set[str]:
    if mode == AccessMode.TOOL:
        return {Visibility.PUBLIC.value, Visibility.GM_ONLY.value, Visibility.TOOL_ONLY.value}
    if mode == AccessMode.GM:
        return {Visibility.PUBLIC.value, Visibility.GM_ONLY.value}
    return {Visibility.PUBLIC.value}


def load_reference_text(
    registry: ContentRegistry,
    *,
    package_id: str,
    reference_id: str,
    mode: AccessMode = AccessMode.GM,
) -> RetrievedSpan:
    package = registry.by_id[package_id]
    for reference in package.manifest.references:
        if reference.id != reference_id:
            continue
        if not can_load_reference(reference, mode):
            raise PermissionError(f"Reference {reference_id} is not visible in {mode} mode")
        path = package.reference_path(reference)
        text = path.read_text(encoding="utf-8")
        return RetrievedSpan(
            package_id=package.id,
            reference_id=reference.id,
            title=reference.title,
            path=str(path),
            text=text,
            visibility=reference.visibility.value,
            score=1,
        )
    raise KeyError(f"Unknown reference {package_id}:{reference_id}")
