import pathlib

import pytest


@pytest.mark.parametrize(
    "args, consumed, rest",
    [
        (["-f", "foo", "bar"], [("-f",)], ["foo", "bar"]),
        (["foo", "bar"], None, ["foo", "bar"]),
    ],
)
def test_match_consume(args, consumed, rest):
    from climate import Option, Match

    @Option.make("-f", "--foo")
    def Foo(cls, mark):
        return True

    assert Match(
        consumed={Foo: consumed} if consumed else {}, remaining=rest
    ) == Foo.consume(*args)


def test_option_make():
    from climate import Option

    @Option.make("-f", "--foo")
    def Foo(cls, _: str) -> Option:
        "Foobar."
        return cls(True)

    assert issubclass(Foo, Option)


def test_parser_match():
    import pathlib
    from climate import Parser, options

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
    parser.add(
        options.ConfigFile,
        options.Verbose,
        options.Debug,
        options.Help,
        options.Path,
        options.Rest,
    )

    parser.match(*args)
    assert parser == Parser(
        consumed={
            options.ConfigFile: [("-c", "config.toml")],
            options.Verbose: [("-vvv",)],
            options.Debug: [("--debug",)],
            options.Help: [("--help",), ("-h",)],
            options.Rest: [("--", "foo", "bar")],
        },
        remaining=[
            "serve",
            "-b",
            "--loop",
            "uvloop",
            "-ca",
            "foobar",
            "--version",
            "-V",
        ],
        values={options.ConfigFile: options.ConfigFile(pathlib.Path("config.toml"))},
        option_marks=[
            ("-c", options.ConfigFile),
            ("--config", options.ConfigFile),
            ("-v", options.Verbose),
            ("-vv", options.Verbose),
            ("-vvv", options.Verbose),
            ("--debug", options.Debug),
            ("-h", options.Help),
            ("--help", options.Help),
            ("-p", options.Path),
            ("--path", options.Path),
            ("--", options.Rest),
        ],
    )


def test_hot_adding_options():
    from climate import CommandLine, Option

    cl = CommandLine()

    @Option.make("foo", eager=True)
    def Foo(cls, _):

        cl.add(Bar)

        return "foo"

    @Option.make("bar")
    def Bar(cls, _, *, foo: Foo):
        return foo.value + "bar"

    cl.add(Foo)
    results = cl.interpret(["foo", "bar"])
    assert results == {Foo: Foo("foo"), Bar: Bar("foobar")}


def test_command_line():
    from climate import CommandLine, Option, options

    @Option.make("foo")
    def Foo(
        cls,
        mark,
        cmd: str,
        *,
        config_file: options.ConfigFile,
        debug: options.Debug,
        verbose: options.Verbose,
        paths: options.Path,
        loop: options.Loop,
    ):

        return {
            "mark": mark,
            "cmd": cmd,
            "config_file": config_file,
            "debug": debug,
            "verbose": verbose,
            "paths": paths,
            "loop": loop,
        }

    cl = CommandLine()
    cl.add(Foo)

    results = cl.interpret(
        [
            "-c",
            "config.toml",
            "--debug",
            "foo",
            "bar",
            "-v",
            "-v",
            "-p",
            "foo.txt",
            "-p",
            "bar.txt",
        ],
    )
    assert results[Foo].value == {
        "mark": "foo",
        "cmd": "bar",
        "config_file": options.ConfigFile(value=pathlib.Path("config.toml")),
        "debug": options.Debug(value=True),
        "paths": options.Path(value=[pathlib.Path("foo.txt"), pathlib.Path("bar.txt")]),
        "verbose": options.Verbose(value=True),
        "loop": options.Loop(None),
    }


@pytest.fixture
def Commands():
    from climate import Group, options

    @Group.make()
    def Commands(cls, member: str, *, config_file: options.ConfigFile):
        return {"member": member, "config_file": config_file}

    @Commands.member()
    def serve(cls, member: str):
        return {"member": member}

    return Commands


@pytest.fixture
def admin(Commands):
    @Commands.member()
    def admin(cls, member: str):
        return {"member": member}

    return admin


@pytest.fixture
def create(admin):
    @admin.member()
    def create(cls, member: str, *, admin: admin):
        return {"member": member, "admin": admin}

    return create


def test_group(Commands, admin, create):
    from climate import options, CommandLine

    cl = CommandLine()
    cl.add(Commands)
    results = cl.interpret(["-c", "config.toml", "admin", "create"])
    assert cl.values, results == (
        {
            options.ConfigFile: options.ConfigFile(pathlib.Path("config.toml")),
            Commands: Commands(
                {
                    "member": "admin",
                    "config_file": options.ConfigFile(pathlib.Path("config.toml")),
                }
            ),
            Commands["admin"]: admin({"member": "create"}),
            create: create({"member": None, "admin": admin({"member": "create"})}),
        },
        {
            options.ConfigFile: options.ConfigFile(pathlib.Path("config.toml")),
            Commands: create({"member": None, "admin": admin({"member": "create"})}),
        },
    )


def _test_help(Commands, admin, create):
    from climate import options, CommandLine

    cl = CommandLine()
    cl.add(options.Help, Commands)
    results = cl.interpret(["-c", "config.toml", "admin", "create", "--help"])
    assert results


def _test_superclass_dependency():
    import typing
    from climate import Option

    @Option.make("-h", "--help", eager=True)
    def Help(cls, _, *, options: typing.List[Option]):
        return True
