"""
Dependency injection of CLI arguments
"""

import inspect
import typing

import attr
import pytest
import typing_inspect


@attr.s
class Match:
    """A Match is a sequence of command line arguments."""

    types: typing.Dict[str, typing.Type] = {}

    def __init_subclass__(cls, *args, **kwargs):
        # only collect Matches, which are implemented
        if cls.serialize is not None:
            cls.types[cls.__name__] = cls

    marks: typing.Tuple[str, ...] = ()
    serialize: typing.Optional[typing.Callable] = None
    nargs: typing.Union[int, bool] = 0
    dependencies: typing.Dict[str, typing.Type["Match"]] = {}

    value: typing.Any = attr.ib()

    @classmethod
    def inspect_serialize(cls, serialize):
        t = inspect.Parameter
        nargs = 0
        dependencies: typing.Dict[str, inspect.Parameter] = {}
        sig = inspect.signature(serialize)
        for name, p in sig.parameters.items():
            if nargs is not True and p.kind in {
                t.POSITIONAL_OR_KEYWORD,
                t.POSITIONAL_ONLY,
            }:
                nargs += 1
            elif p.kind is t.VAR_POSITIONAL:
                nargs = True
            elif p.kind is t.VAR_KEYWORD:
                raise SyntaxError(
                    f"Variable keyword argument makes no sense for a `Match`: {name}",
                    sig,
                    serialize,
                )
            else:
                dependencies[name] = p

        # we decrease for the fact, that we require cls as the first argument
        nargs -= 1
        if nargs == 0:
            raise SyntaxError(
                (
                    "You need to define at least two positional arguments"
                    " one for the class and one for the mark."
                ),
                sig,
                serialize,
            )

        return nargs, dependencies

    @classmethod
    def options(cls):
        def _():
            for option in cls.types.values():
                for mark in option.marks:
                    yield mark, option

        return sorted(_())

    @classmethod
    def make(cls, *marks: str, help: str = None):
        def decorator(func):
            name = func.__name__
            nargs, dependencies = cls.inspect_serialize(func)
            attributes = {
                "marks": marks,
                "serialize": classmethod(func),
                "nargs": nargs,
                "dependencies": dependencies,
                "__doc__": help or func.__doc__,
                "__repr__": lambda self: f"<{name}: {' '.join(marks)}>",
            }
            option_cls = type.__new__(type, name, (cls,), attributes)
            return option_cls

        return decorator

    @classmethod
    def consume(cls, *args):
        if not any(mark.startswith(args[0]) for mark in cls.marks):
            return (), args
        if cls.nargs is True:
            return args, ()

        return args[0 : cls.nargs], args[cls.nargs :]


@attr.s
class Option(Match, skip=True):
    @classmethod
    def match(cls, *arguments):
        # we sort our options by marks
        options = cls.options()
        mismatched: typing.Dict[typing.Tuple, typing.Set] = {}
        consumed = {}
        remaining = []

        while arguments:
            marks, option_cls = options.pop(0)

            # we already tried this combination
            if arguments in mismatched and marks in mismatched[arguments]:
                remaining.append(arguments[0])
                arguments = arguments[1:]
                # reset option
                options.append((marks, option_cls))
                continue

            # skip already consumed options
            if option_cls in consumed:
                continue

            consumed_arguments, arguments = option_cls.consume(*arguments)
            if consumed_arguments:
                consumed[option_cls] = consumed_arguments
            else:
                # reset option
                options.append((marks, option_cls))
                # remember failed option for arguments
                mismatched.setdefault(arguments, set()).add(marks)

        return consumed, remaining


@Option.make("-c", "--config")
def ConfigFile(cls, _, path: str):
    "The config file."
    return cls(str(path))


@Option.make("-p", "--path")
def Path(cls, _, path: str):
    "Project path."
    import pathlib

    return cls(value=pathlib.Path(path))


@Match.make("--debug")
def Debug(cls, _: str):
    "Enable debug."
    return cls(True)


@Match.make("--de")
def De(cls, argument: str):
    "Enable debug."
    return cls(True)


@Match.make("-vvvvv")
def Verbose(cls, argument: str):
    "Enable verbosity."
    return cls(True)


@Match.make("-h", "--help")
def Help(cls, argument: str):
    "Show help."
    return cls(True)


##################################################


@pytest.mark.parametrize(
    "args, consumed, rest",
    [
        (["-f", "foo", "bar"], ("-f",), ("foo", "bar")),
        (["foo", "bar"], (), ("foo", "bar")),
    ],
)
def test_match_consume(args, consumed, rest):
    @Match.make("-f", "--foo")
    def Foo(cls, mark):
        return True

    assert (consumed, rest) == Foo.consume(*args)


def test_option_make():
    @Option.make("-f", "--foo")
    def Foo(cls, _: str) -> Option:
        "Foobar."
        return cls(True)

    assert issubclass(Foo, Match)
    assert issubclass(Foo, Option)


def test_option_consume():
    args = [
        "-c",
        "config.toml",
        "serve",
        "-b",
        "--loop",
        "uvloop",
        "-vvv",
        "-ca",
        "foobar",
        "--debug",
        "--help",
        "-h",
        "--version",
        "-V",
    ]
    consumed, remaining = Option.match(*args)
    assert consumed == {
        ConfigFile: ("-c", "config.toml"),
        Verbose: ("-vvv",),
        Debug: ("--debug",),
        Help: ("--help",),
    }
    assert remaining == [
        "serve",
        "-b",
        "--loop",
        "uvloop",
        "-ca",
        "foobar",
        # XXX remove -h
        "-h",
        "--version",
        "-V",
    ]
