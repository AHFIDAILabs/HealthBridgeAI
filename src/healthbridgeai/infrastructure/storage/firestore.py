"""FirestoreUserStore / FirestoreConversationStore — IUserStore + IConversationStore."""
from __future__ import annotations

import hashlib
import time
from typing import Optional

import structlog
from google.cloud import firestore
from google.cloud.firestore_v1.async_client import AsyncClient

from healthbridgeai.config.settings import settings
from healthbridgeai.core.models.user import ConversationTurn, RateLimit, User

log = structlog.get_logger(__name__)

_USERS_COL = "users"
_RATE_COL = "rate_limits"
_CONV_COL = "conversations"
_TURNS_SUB = "turns"
_MAX_TURNS_STORED = 40  # cap per user; only last N returned


def _db() -> AsyncClient:
    """Lazy singleton Firestore client (process-level)."""
    if not hasattr(_db, "_client"):
        _db._client = firestore.AsyncClient(
            project=settings.GCP_PROJECT_ID,
            database=settings.FIRESTORE_DATABASE,
        )
    return _db._client


def _phone_hash(phone_number: str) -> str:
    return hashlib.sha256(phone_number.encode()).hexdigest()[:12]


# ── IUserStore ────────────────────────────────────────────────────────────────

class FirestoreUserStore:
    """Stores User documents keyed by phone_hash."""

    async def get_user(self, phone_number: str) -> Optional[User]:
        doc_id = _phone_hash(phone_number)
        snap = await _db().collection(_USERS_COL).document(doc_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        data["phone_number"] = phone_number  # not stored in Firestore; inject at read time
        try:
            return User(**data)
        except Exception as exc:
            log.error("firestore.get_user.parse_error", error=str(exc), doc_id=doc_id)
            return None

    async def upsert_user(self, user: User) -> None:
        doc_id = _phone_hash(user.phone_number)
        data = user.model_dump(exclude={"phone_number"})  # never persist plain phone number
        data.pop("phone_hash", None)  # computed field; not stored
        await _db().collection(_USERS_COL).document(doc_id).set(data, merge=True)

    async def check_rate_limit(
        self, phone_hash: str, limit: int, window_seconds: int
    ) -> RateLimit:
        doc_ref = _db().collection(_RATE_COL).document(phone_hash)

        @firestore.async_transactional
        async def _txn(transaction, ref) -> RateLimit:
            snap = await ref.get(transaction=transaction)
            now = int(time.time())
            if snap.exists:
                d = snap.to_dict()
                window_start = d.get("window_start", now)
                count = d.get("message_count", 0)
                if now - window_start >= window_seconds:
                    # New window
                    window_start = now
                    count = 1
                else:
                    count += 1
            else:
                window_start = now
                count = 1

            transaction.set(
                ref,
                {"phone_hash": phone_hash, "window_start": window_start, "message_count": count},
                merge=False,
            )
            return RateLimit(
                phone_hash=phone_hash,
                window_start=window_start,
                message_count=count,
            )

        transaction = _db().transaction()
        return await _txn(transaction, doc_ref)


# ── IConversationStore ────────────────────────────────────────────────────────

class FirestoreConversationStore:
    """Stores conversation turns as Firestore sub-documents under conversations/{hash}/turns."""

    async def get_recent_turns(self, phone_hash: str, n: int = 5) -> list[ConversationTurn]:
        col = (
            _db()
            .collection(_CONV_COL)
            .document(phone_hash)
            .collection(_TURNS_SUB)
        )
        query = col.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(n)
        snaps = await query.get()
        turns = []
        for snap in reversed(snaps):  # oldest first
            try:
                turns.append(ConversationTurn(**snap.to_dict()))
            except Exception as exc:
                log.warning("firestore.get_turns.parse_error", error=str(exc))
        return turns

    async def save_turn(self, phone_hash: str, turn: ConversationTurn) -> None:
        col = (
            _db()
            .collection(_CONV_COL)
            .document(phone_hash)
            .collection(_TURNS_SUB)
        )
        # Use timestamp as document ID to enable ordering without Firestore index on a field
        doc_id = str(turn.timestamp) + "_" + turn.role[:1]
        await col.document(doc_id).set(turn.model_dump())
        # Best-effort trim: if we have too many turns, delete the oldest ones
        await self._trim(col)

    async def clear_history(self, phone_hash: str) -> None:
        col = (
            _db()
            .collection(_CONV_COL)
            .document(phone_hash)
            .collection(_TURNS_SUB)
        )
        snaps = await col.get()
        for snap in snaps:
            await snap.reference.delete()

    async def _trim(self, col) -> None:
        try:
            all_snaps = await col.order_by("timestamp").get()
            excess = len(all_snaps) - _MAX_TURNS_STORED
            if excess > 0:
                for snap in all_snaps[:excess]:
                    await snap.reference.delete()
        except Exception:
            pass  # trim is best-effort
