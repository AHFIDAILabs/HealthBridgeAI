"""IUserStore / IConversationStore — contracts for Firestore persistence."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..models.user import ConversationTurn, RateLimit, User


@runtime_checkable
class IUserStore(Protocol):
    async def get_user(self, phone_number: str) -> Optional[User]:
        """Return user document or None if first-time contact."""
        ...

    async def upsert_user(self, user: User) -> None:
        """Create or fully overwrite the user document."""
        ...

    async def check_rate_limit(self, phone_hash: str, limit: int, window_seconds: int) -> RateLimit:
        """
        Atomic increment of message count in the current window.
        Returns the updated RateLimit — caller raises RateLimitError if exceeded.
        """
        ...


@runtime_checkable
class IConversationStore(Protocol):
    async def get_recent_turns(self, phone_hash: str, n: int = 5) -> list[ConversationTurn]:
        """Return the last n turns, oldest first, for LLM context injection."""
        ...

    async def save_turn(self, phone_hash: str, turn: ConversationTurn) -> None:
        """Append a turn. Implementations should cap history at a max length (e.g. 20 turns)."""
        ...

    async def clear_history(self, phone_hash: str) -> None:
        """Delete all conversation history for a user (on explicit user request)."""
        ...
