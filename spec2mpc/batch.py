from dataclasses import dataclass, fields, replace
from typing import Any, Sequence, Type, TypeVar, Union

import torch
from torch import Tensor

T = TypeVar("T", bound="Batched")


@dataclass
class Batched:
    """Dataclass mixin: every field is a tensor with leading batch dim B."""

    @property
    def batch_size(self) -> int:
        return int(getattr(self, fields(self)[0].name).shape[0])

    def index(self: T, idx: Union[Tensor, slice]) -> T:
        # idx: slice or long tensor [M] -> every field [M, ...]
        return replace(self, **{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    def to(self: T, device: Any) -> T:
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    @classmethod
    def cat(cls: Type[T], items: Sequence[T]) -> T:
        # items: list of [B_i, ...] -> [sum B_i, ...]; dtype preserved per field
        return cls(**{f.name: torch.cat([getattr(i, f.name) for i in items], dim=0) for f in fields(cls)})
