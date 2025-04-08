import enum
from typing import Sequence, TypeVar

T = TypeVar("T", bound=enum.Enum)

class StrEnumMeta(enum.EnumMeta):
    auto = enum.auto
    def from_str(self, member: str) -> T: ...

class StrEnum(enum.Enum, metaclass=StrEnumMeta): ...

def sequence_to_str(seq: Sequence, separate_last: str = "") -> str: ...
