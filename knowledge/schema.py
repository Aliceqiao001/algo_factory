"""Dataclass definitions for all knowledge graph entities: capabilities, schemas, validation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HyperParameter:
    """A single tunable hyperparameter for an algorithm."""

    name: str
    default: Any
    range: Optional[List[Any]]
    description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "default": self.default,
            "range": self.range,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HyperParameter":
        return cls(
            name=d["name"],
            default=d["default"],
            range=d.get("range"),
            description=d.get("description", ""),
        )


@dataclass
class ValidationRecord:
    """One execution result recorded after running a capability in the sandbox."""

    timestamp: str
    success: bool
    metrics: Dict[str, float]
    error_message: Optional[str]
    code_version: str
    notes: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "success": self.success,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "code_version": self.code_version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationRecord":
        return cls(
            timestamp=d["timestamp"],
            success=d["success"],
            metrics=d.get("metrics", {}),
            error_message=d.get("error_message"),
            code_version=d.get("code_version", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class InputSchema:
    """Describes the expected shape of data fed into an algorithm."""

    required_columns: List[str]
    optional_columns: List[str]
    min_rows: int
    data_types: Dict[str, str]

    def to_dict(self) -> dict:
        return {
            "required_columns": self.required_columns,
            "optional_columns": self.optional_columns,
            "min_rows": self.min_rows,
            "data_types": self.data_types,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InputSchema":
        return cls(
            required_columns=d.get("required_columns", []),
            optional_columns=d.get("optional_columns", []),
            min_rows=d.get("min_rows", 0),
            data_types=d.get("data_types", {}),
        )


@dataclass
class OutputSchema:
    """Describes what an algorithm produces."""

    format: str
    columns: List[str]
    description: str

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "columns": self.columns,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OutputSchema":
        return cls(
            format=d.get("format", ""),
            columns=d.get("columns", []),
            description=d.get("description", ""),
        )


@dataclass
class AlgorithmCapability:
    """A complete description of one reusable algorithm capability node in the knowledge graph."""

    id: str
    name: str
    category: str
    description: str
    applicable_conditions: List[str]
    input_schema: InputSchema
    output_schema: OutputSchema
    metrics: List[str]
    dependencies: List[str]
    code_template: str
    hyperparameters: List[HyperParameter] = field(default_factory=list)
    validation_history: List[ValidationRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "applicable_conditions": self.applicable_conditions,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "metrics": self.metrics,
            "dependencies": self.dependencies,
            "code_template": self.code_template,
            "hyperparameters": [h.to_dict() for h in self.hyperparameters],
            "validation_history": [v.to_dict() for v in self.validation_history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AlgorithmCapability":
        return cls(
            id=d["id"],
            name=d["name"],
            category=d.get("category", ""),
            description=d.get("description", ""),
            applicable_conditions=d.get("applicable_conditions", []),
            input_schema=InputSchema.from_dict(d.get("input_schema", {})),
            output_schema=OutputSchema.from_dict(d.get("output_schema", {})),
            metrics=d.get("metrics", []),
            dependencies=d.get("dependencies", []),
            code_template=d.get("code_template", ""),
            hyperparameters=[HyperParameter.from_dict(h) for h in d.get("hyperparameters", [])],
            validation_history=[ValidationRecord.from_dict(v) for v in d.get("validation_history", [])],
        )


@dataclass
class DataPattern:
    """A named pattern describing a dataset characteristic that influences algorithm selection."""

    id: str
    name: str
    description: str

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description}

    @classmethod
    def from_dict(cls, d: dict) -> "DataPattern":
        return cls(id=d["id"], name=d["name"], description=d.get("description", ""))


@dataclass
class FailureCase:
    """A documented failure mode with a known fix strategy, linked to a capability."""

    id: str
    error_type: str
    error_message: str
    fix_strategy: str
    related_capability_id: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "fix_strategy": self.fix_strategy,
            "related_capability_id": self.related_capability_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailureCase":
        return cls(
            id=d["id"],
            error_type=d.get("error_type", ""),
            error_message=d.get("error_message", ""),
            fix_strategy=d.get("fix_strategy", ""),
            related_capability_id=d.get("related_capability_id", ""),
        )
