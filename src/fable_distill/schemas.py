from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .io import sha256_json


class TargetType(str, Enum):
    NEXT_ACTION = "next_action"
    TRAJECTORY = "trajectory"
    FINAL_ANSWER = "final_answer"
    TOOL_CALL = "tool_call"


class ReasoningMode(str, Enum):
    LONG = "long"
    COMPRESSED = "compressed"
    HIDDEN = "hidden"


VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(slots=True)
class Message:
    role: str
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        self.role = self.role.lower().strip()
        if self.role not in VALID_ROLES:
            raise ValueError(f"Unsupported message role: {self.role!r}")
        if not isinstance(self.content, str):
            self.content = str(self.content)
        if self.role == "tool" and self.name is not None:
            self.name = str(self.name)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Message":
        return cls(
            role=str(value.get("role", "")),
            content=str(value.get("content", "")),
            name=value.get("name"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass(slots=True)
class Provenance:
    source_dataset: str
    source_revision: str = "unknown"
    source_session_id: str = "unknown"
    source_example_id: str = "unknown"
    license: str = "unknown"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Provenance":
        return cls(
            source_dataset=str(value.get("source_dataset", "unknown")),
            source_revision=str(value.get("source_revision", "unknown")),
            source_session_id=str(value.get("source_session_id", "unknown")),
            source_example_id=str(value.get("source_example_id", "unknown")),
            license=str(value.get("license", "unknown")),
        )


@dataclass(slots=True)
class CanonicalExample:
    messages: list[Message]
    provenance: Provenance
    session_id: str
    repository_id: str
    task_family: str = "unknown"
    language: str = "unknown"
    target_type: str = TargetType.NEXT_ACTION.value
    reasoning_mode: str = ReasoningMode.COMPRESSED.value
    verified_success: bool = False
    quality_score: float = 0.0
    example_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("A canonical example must contain messages")
        if not any(message.role == "assistant" for message in self.messages):
            raise ValueError("A canonical example needs at least one assistant target")
        if self.target_type not in {item.value for item in TargetType}:
            raise ValueError(f"Invalid target_type: {self.target_type}")
        if self.reasoning_mode not in {item.value for item in ReasoningMode}:
            raise ValueError(f"Invalid reasoning_mode: {self.reasoning_mode}")
        self.quality_score = max(0.0, min(1.0, float(self.quality_score)))
        if not self.example_id:
            self.example_id = self.compute_id()

    def compute_id(self) -> str:
        identity = {
            "messages": [message.to_dict() for message in self.messages],
            "source_dataset": self.provenance.source_dataset,
            "source_revision": self.provenance.source_revision,
            "session_id": self.session_id,
            "target_type": self.target_type,
        }
        return sha256_json(identity)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "example_id": self.example_id,
            "session_id": self.session_id,
            "repository_id": self.repository_id,
            "task_family": self.task_family,
            "language": self.language,
            "messages": [message.to_dict() for message in self.messages],
            "target_type": self.target_type,
            "reasoning_mode": self.reasoning_mode,
            "verified_success": self.verified_success,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }
        result.update(asdict(self.provenance))
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalExample":
        provenance_value = value.get("provenance") or value
        return cls(
            example_id=str(value.get("example_id", "")),
            session_id=str(value.get("session_id", "unknown")),
            repository_id=str(value.get("repository_id", "unknown")),
            task_family=str(value.get("task_family", "unknown")),
            language=str(value.get("language", "unknown")),
            provenance=Provenance.from_dict(provenance_value),
            messages=[Message.from_dict(item) for item in value.get("messages", [])],
            target_type=str(value.get("target_type", TargetType.NEXT_ACTION.value)),
            reasoning_mode=str(value.get("reasoning_mode", ReasoningMode.COMPRESSED.value)),
            verified_success=bool(value.get("verified_success", False)),
            quality_score=float(value.get("quality_score", 0.0)),
            metadata=dict(value.get("metadata") or {}),
        )

