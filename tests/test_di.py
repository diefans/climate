"""
Dependency injection of CLI arguments
"""
import bisect
import functools
import inspect
import itertools
import operator
import typing

import attr
import pytest
import typing_inspect


@attr.s
class Option:
    """A Option is a sequence of command line arguments bound to a starter mark."""

    marks: typing.Tuple[str, ...] = ()
    serialize: typing.Optional[typing.Callable] = None
    nargs: typing.Union[int, bool] = 0
    dependencies: typing.Dict[str, typing.Type["Option"]] = {}

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
                    f"Variable keyword argument makes no sense for a `Option`: {name}",
                    sig,
                    serialize,
                )
            else:
                dependencies[name] = p

        # we decrease for the fact, that we require cls as the first argument
        if nargs is not True:
            nargs -= 1
        if nargs == 0:
            breakpoint()  # XXX BREAKPOINT
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
        if not any(map(functools.partial(operator.eq, args[0]), cls.marks)):
            return (), args
        if cls.nargs is True:
            return args, ()

        return args[0 : cls.nargs], args[cls.nargs :]


@attr.s
class Parser:
    options: typing.List[typing.Tuple[str, Option]] = attr.ib(factory=list)

    def add(self, option: Option, *options: Option):
        # add all marks sorted
        # XXX does it make sense to sort???
        # we could use a dict or a set to avoid lookup for duplicates
        for m in itertools.chain((option,), options):
            for mark in m.marks:
                bisect.insort(self.options, (mark, m))

    def match(self, *arguments):
        options = self.options
        consumed: typing.Dict[Option, typing.List[typing.Tuple[str, ...]]] = {}
        remaining: typing.List[str] = []
        failed: typing.Dict[typing.Tuple, typing.Set] = {}

        while arguments:
            mark, option_cls = options.pop(0)

            # we already tried this combination
            if arguments in failed and mark in failed[arguments]:
                remaining.append(arguments[0])
                arguments = arguments[1:]
            else:
                consumed_arguments, arguments = option_cls.consume(*arguments)
                if consumed_arguments:
                    consumed.setdefault(option_cls, list()).append(consumed_arguments)
                else:
                    # remember failed option for arguments
                    failed.setdefault(arguments, set()).add(mark)

            # reset option
            options.append((mark, option_cls))
        return consumed, remaining


@attr.s
class CommandLine:
    parser: Parser = attr.ib(factory=Parser)

    def add(self, option: Option):
        self.parser.add(option)
        return option

    def __call__(self, *args):
        # search for Option dependencies in the parser

        consumed = {
            option_cls: list(map(operator.itemgetter(1), group))
            for option_cls, group in itertools.groupby(
                self.parser.match(*args), operator.itemgetter(0)
            )
        }
        return consumed


@Option.make("-c", "--config")
def ConfigFile(cls, _, path: str):
    "The config file."
    return cls(str(path))


@Option.make("-p", "--path")
def Path(cls, _, path: str):
    "Project path."
    import pathlib

    return cls(value=pathlib.Path(path))


@Option.make("--debug")
def Debug(cls, _: str):
    "Enable debug."
    return cls(True)


@Option.make("--de")
def De(cls, argument: str):
    "Enable debug."
    return cls(True)


@Option.make("-v", "-vv", "-vvv")
def Verbose(cls, argument: str):
    "Enable verbosity."
    return cls(True)


@Option.make("-h", "--help")
def Help(cls, argument: str):
    "Show help."
    return cls(True)


@Option.make("--")
def Rest(cls, *args):
    return


##################################################


@pytest.mark.parametrize(
    "args, consumed, rest",
    [
        (["-f", "foo", "bar"], ("-f",), ("foo", "bar")),
        (["foo", "bar"], (), ("foo", "bar")),
    ],
)
def test_match_consume(args, consumed, rest):
    @Option.make("-f", "--foo")
    def Foo(cls, mark):
        return True

    assert (consumed, rest) == Foo.consume(*args)


def test_option_make():
    @Option.make("-f", "--foo")
    def Foo(cls, _: str) -> Option:
        "Foobar."
        return cls(True)

    assert issubclass(Foo, Option)


def test_command_line_consume():
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
        "--",
        "foo",
        "bar",
    ]
    parser = Parser()
    parser.add(ConfigFile, Verbose, Debug, Help, Path, Rest)

    consumed, remaining = parser.match(*args)
    assert consumed == {
        ConfigFile: [("-c", "config.toml")],
        Verbose: [("-vvv",)],
        Debug: [("--debug",)],
        Help: [("--help",), ("-h",)],
        Rest: [("--", "foo", "bar")],
    }
    assert remaining == [
        "serve",
        "-b",
        "--loop",
        "uvloop",
        "-ca",
        "foobar",
        "--version",
        "-V",
    ]
