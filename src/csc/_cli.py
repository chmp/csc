import ast
import argparse
import warnings

from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

from ._script import FileSource, InlineSource, DEFAULT_CELL_MARKER, Script


_parser = argparse.ArgumentParser("csc")
_parser.add_argument(
    "-p,--param",
    action="append",
    dest="parameters",
)
_parser.add_argument("--cell-marker", default=DEFAULT_CELL_MARKER)
_parser.add_argument("--register", action="store_true", default=False)
_parser.add_argument("files", nargs="+")


def main(args: None | Sequence[str] = None) -> Script:
    parsed_args = parse_args(args)
    script = build_script(parsed_args)
    script.run(*script.list())
    return script


@dataclass
class Arguments:
    cell_marker: str
    register: bool
    files: tuple[Path, ...]
    parameters: tuple[str, ...]


def parse_args(args: None | Sequence[str] = None) -> Arguments:
    raw_args = _parser.parse_args(args)

    files = tuple(Path(p) for p in raw_args.files)

    if raw_args.parameters is not None:
        parameters = tuple(p for p in raw_args.parameters)
    else:
        parameters = ()

    return Arguments(
        files=files,
        parameters=parameters,
        cell_marker=raw_args.cell_marker,
        register=raw_args.register,
    )


def build_script(args: Arguments) -> Script:
    sources = []
    for file in args.files:
        sources.append(FileSource(file, cell_marker=args.cell_marker))

    if args.parameters:
        sources.append(build_parameter_source(args.parameters))

    script = Script(sources, register=args.register, cell_marker=args.cell_marker)

    if args.parameters and "parameters" not in script.list():
        warnings.warn(
            "CLI called with parameters, but the script has no 'parameters' cell"
        )

    return script


def build_parameter_source(parameters: tuple[str, ...]):
    # TODO: parse parameters and generate a guard for each parameter to check
    # that it was defined

    assigned_names = get_assigned_names("\n".join(parameters))

    lines = [
        f"{DEFAULT_CELL_MARKER} parameters",
        *(build_parameter_guard(name) for name in assigned_names),
        *parameters,
    ]

    return InlineSource("\n".join(lines), cell_marker=DEFAULT_CELL_MARKER)


def get_assigned_names(source: str) -> tuple[str, ...]:
    names: list[str] = []

    mod = ast.parse(source)
    for statement in mod.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)

    return tuple(names)


def build_parameter_guard(name: str) -> str:
    return (
        "try:\n"
        f"    {name}\n"
        "except NameError:\n"
        f"    __import__('warnings').warn('Parameter {name} is not defined in base script, but overwritten')"
    )
