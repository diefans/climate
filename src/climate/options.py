import pathlib

from . import Option, Match


@Option.make("-c", "--config", eager=True)
def ConfigFile(cls, _, path: str):
    "The config file."
    return pathlib.Path(path)


@Option.make("-p", "--path", particular=None)
def Path(cls, _, paths: str):
    "Project path."
    return [pathlib.Path(path) for path in paths]


class Flag:
    @classmethod
    def serialize(cls, _: str):
        return True


class Debug(Option, Flag, marks=("--debug",), particular=None):
    ...


class Verbose(Option, Flag, marks=("-v", "-vv", "-vvv"), particular=None):
    ...


class Help(
    Option, Flag, marks=("-h", "--help"), particular=None, exclusive=True, default=None
):
    """Describe available options and their impact."""

    @classmethod
    def inject(cls, match: Match):
        options = []
        # collect all registered option hekp sections

        # get the widest marks to define left column width
        # we calculate the golden cut for the left column
        # if the calculated width is less the minimal width
        # we take the minimal width and calculate the golden cut from that
        # if the title is wider, we take the golden cut from that
        # 38.2% vs. 61.8%

        ...


@Option.make("--loop", default=None)
def Loop(cls, _, loop_name: str):
    return loop_name


@Option.make("--")
def Rest(cls, _, *args):
    return args
