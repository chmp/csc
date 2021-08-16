from os import PathLike
from typing import Optional, Sequence, TextIO, Union


def read(s: Union[str, PathLike, TextIO], as_version: Optional[int] = None) -> 'TypedNotebook':
    ...


class NotebookNode:
    pass


class TypedNotebook(NotebookNode):
    cells: Sequence['TypedCell']


class TypedCell(NotebookNode):
    cell_type: str
    source: str
