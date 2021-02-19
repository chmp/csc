"""A repl to interactively run the script"""
import json
import traceback
from enum import Enum, auto

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


def repl(script, widget=False):
    if widget is True:
        return _widget_repl(script)

    return _cli_repl(script)


def _cli_repl(script):
    command_banner(script, "/banner")

    running = True
    while running:
        inp = input()
        running = _repl_step(script, inp)


def _widget_repl(script):
    from ipywidgets import VBox, Text, Output, Layout
    from IPython.display import display

    user_input = Text(placeholder="Execute cell / command")
    output = Output()
    layout = Layout(border="solid 1px #555")
    repl = VBox([output, user_input], layout=layout)

    @user_input.on_submit
    def _(slf, *_):
        inp = slf.value
        slf.value = ""

        # Ensure to only accept inputs after the last command was evaluated
        slf.disabled = True

        try:
            with output:
                _repl_step(script, inp)
        finally:
            slf.disabled = False

    with output:
        command_banner(script, "/banner")

    display(repl)


def _repl_step(script, inp):
    if script._repl.get("autoclear"):
        clear_output()
        print(inp, flush=True)

    if not inp.strip():
        return True

    for cmd in _repl_commands:
        cmd_res = cmd(script, inp)

        if cmd_res is None:
            break

        elif cmd_res is Result.quit:
            return False

        elif cmd_res is Result.no_match:
            pass

        else:
            print(f"Invalid command result: {cmd_res!r}")

    else:
        print("Did not understand command")

    print()
    return True


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
def command_banner(script, inp):
    if inp.strip() != "/banner":
        return Result.no_match

    print(repl_banner.format(name=script.path.name))
    print("available cells:", *script.list())
    print()


@repl_command
def command_help(script, inp):
    if inp.strip() not in {"/help"}:
        return Result.no_match

    print(repl_help)


@repl_command
def command_list(script, inp):
    if inp.strip() not in {"/list"}:
        return Result.no_match

    print("available cells:", *script.list())


@repl_command
def command_clear(script, inp):
    if inp.strip() not in {"/clear"}:
        return Result.no_match

    clear_output()


@repl_command
def command_who(script, inp):
    if inp.strip() not in {"/who"}:
        return Result.no_match

    print([k for k in vars(script.ns) if not k.startswith("_")])


@repl_command
def command_eval(script, inp):
    inp = inp.strip()
    if not inp.startswith("/eval "):
        return Result.no_match

    code = inp[len("/eval ") :]
    try:
        res = script.eval(code)
        display(res)
        show_figures()

    except:
        traceback.print_exc()


@repl_command
def command_exec(script, inp):
    inp = inp.strip()
    if not inp.startswith("/exec"):
        return Result.no_match

    inp = inp[len("/exec") :].strip()

    if inp == "":
        code = []
        while True:
            line = input()
            if not line.strip():
                break

            code += [line]

        code = "\n".join(code)

    else:
        code = inp

    try:
        script.exec(code)
        show_figures()

    except:
        traceback.print_exc()


@repl_command
def command_run_all(script, inp):
    inp = inp.strip()
    if not inp.startswith("/run"):
        return Result.no_match

    cells = inp[len("/run") :].strip()

    # TODO:support running a list of cells

    try:
        if cells == "":
            script.run_all()

        else:
            cells = json.loads(f"[{cells}]")
            script.run(*cells)

        show_figures()

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
    evaluate an expression in the script context and show the result
/exec {statement}
    evalute a single statement
/exec
    evaluate multi-line statements, end with with an empty line
/run
    run all cells in the script
/run "cell 1", "cell 2"
    run the given cells
/who
    show variables defined in the script namespace
/banner
    show the initial banner
"""[
    1:-1
]
