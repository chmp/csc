"""A repl to interactively run the script"""
from enum import Enum, auto
import traceback

try:
    from IPython.display import clear_output, display

except ImportError:

    def clear_output():
        pass

    def display(obj):
        if obj is not None:
            print(obj)


try:
    import matplotlib.pyplot as plt

    show_figures = plt.show

except ImportError:

    def show_figures():
        pass


def repl(script):
    print(repl_banner.format(name=script.path.name))

    while True:
        print()
        inp = input()

        for cmd in _repl_commands:
            cmd_res = cmd(script, inp)

            if cmd_res is None:
                break

            elif cmd_res is Result.quit:
                return

            elif cmd_res is Result.no_match:
                continue

            else:
                print(f"Invalid command result: {cmd_res!r}")

        else:
            print("Did not understand command")


_repl_commands = []


def repl_command(func):
    _repl_commands.append(func)
    return func


@repl_command
def command_quit(script, inp):
    if inp.strip() not in {"/quit", "/q"}:
        return Result.no_match

    return Result.quit


@repl_command
def command_help(script, inp):
    if inp.strip() not in {"/help"}:
        return Result.no_match

    print(repl_help)


@repl_command
def command_list(script, inp):
    if inp.strip() not in {"/list"}:
        return Result.no_match

    print(script.list())


@repl_command
def command_clear(script, inp):
    if inp.strip() not in {"/clear"}:
        return Result.no_match

    clear_output()


@repl_command
def command_eval(script, inp):
    inp = inp.strip()
    if not inp.startswith("/eval "):
        return Result.no_match

    code = inp[len("/eval ") :]
    try:
        res = script.eval(code)
        display(res)

    except:
        traceback.print_exc()


@repl_command
def command_execute_cell(script, inp):
    inp = inp.strip()
    if inp.startswith("/"):
        return Result.no_match

    try:
        script.run(inp)
        show_figures()

    except:
        traceback.print_exc()


class Result(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    no_match = auto()
    quit = auto()


# generated wiht https://patorjk.com/software/taag/
repl_banner = r"""
   ___ ____ ___ 
  / __/ __/ __/
 | (__\__\ (___
  \___/___\___/ {name}
  
Type /help for available command and /quit to exit
"""[
    1:-1
]

repl_help = r"""
To run a cell simply type its name (or the beginning of its names). In addition 
the following commands are available:

/quit | /q
    quit the REPL
/help
    show this help
/list
    list available cells
/clear
    clear the output
/eval
    evaluate a statement in the script context and show the result
"""[
    1:-1
]
