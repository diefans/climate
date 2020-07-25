"""
Dependency injection of CLI arguments
"""
import functools
import inspect
import itertools
import operator
import sys
import typing

import attr

OptionType = typing.Type["Option"]
GroupType = typing.Type["Group"]
Consumed = typing.Dict[
    typing.Type["Option"], typing.List[typing.Tuple[typing.Optional[str], ...]]
]
Arguments = typing.List[str]
Values = typing.Dict[OptionType, "Option"]


@functools.partial(lambda x: x())
class missing:
    def __repr__(self):
        return self.__class__.__name__

    def __bool__(self):
        return False


class ClimateException(Exception):
    ...


class MissingOption(ClimateException):
    ...


class EagerDependencyConflict(ClimateException):
    ...


class MultipleOptions(ClimateException):
    ...


def single(first=False, last=False):
    """Restrict arguments to one value instead of a list."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(cls, *args, **kwargs):
            if len(args[0]) == 1 or first:
                # pass reduced args
                return func(cls, *next(zip(*args)), **kwargs)
            if last:
                # pass reversed reduced args
                return func(cls, *next(reversed(iter(zip(*args)), **kwargs)))
            raise MultipleOptions(cls, args)

        return wrapper

    return decorator


# XXX TODO test for Option subclass
def interpret_nargs_dependencies(func):
    t = inspect.Parameter
    nargs = 0
    dependencies: typing.Dict[str, typing.Type["Option"]] = {}
    sig = inspect.signature(func)
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
                func,
            )
        else:
            assert issubclass(p.annotation, Option)
            dependencies[name] = p.annotation

    if nargs == 0:
        raise SyntaxError(
            (
                "You need to define at least two positional arguments"
                " one for the class and one for the mark."
            ),
            sig,
            func,
        )

    return nargs, dependencies


@attr.s
class Match:
    consumed: Consumed = attr.ib(factory=dict)
    # XXX TODO remaining should not be part of Match
    remaining: typing.List[str] = attr.ib(factory=list)
    values: typing.Dict[OptionType, "Option"] = attr.ib(factory=dict)

    @property
    def option_args(
        self,
    ) -> typing.Dict[typing.Type["Option"], typing.List[typing.Tuple]]:
        option_args = {
            option_cls: list(zip(*values))
            for option_cls, values in self.consumed.items()
        }
        return option_args

    def add_option_args(self, cls, args):
        self.consumed.setdefault(cls, []).append(args)

    def join(self, match):
        for option_cls, consumed in match.consumed.items():
            self.consumed.setdefault(option_cls, []).extend(consumed)


@attr.s
class Parser(Match):
    # order defines the order of help
    option_marks: typing.List[typing.Tuple[typing.Optional[str], OptionType]] = attr.ib(
        factory=list
    )

    @property
    def options(self):
        return {option_cls for _, option_cls in self.option_marks}

    def add(self, *options: OptionType):
        for option_cls in options:
            # for mark in option_cls.marks:
            #     option_mark = (mark, option_cls)
            #     if option_mark not in self.option_marks:
            self.option_marks.extend(
                option_mark
                for option_mark in option_cls.option_marks()
                if option_mark not in self.option_marks
            )

    def match(self, *arguments: str):
        while arguments:
            option_marks = list(self.option_marks)
            while option_marks:
                mark, option_cls = option_marks.pop(0)
                # add dependencies
                # this is more expensive than loading dependencies when adding options,
                # but we keep a clean list of user provided options in self.option_marks
                option_marks.extend(option_cls.all_dependencies())

                match = option_cls.consume(*arguments)
                if match.consumed:
                    self.join(match)
                    arguments = match.remaining
                    if option_cls.eager:
                        option_cls.inject(self)
                    break
            else:
                self.remaining.append(arguments[0])
                arguments = arguments[1:]

    def generate_all_options(self):
        options = set(self.option_marks)
        for _, option_cls in list(options):
            options.update(
                itertools.chain(
                    *(o.option_marks() for _, o in option_cls.all_dependencies())
                )
            )

        breakpoint()  # XXX BREAKPOINT
        # for option_cls in


@attr.s
class CommandLine(Parser):
    def interpret(self, args: typing.List[str] = sys.argv[1:]):
        self.match(*args)
        # run exclusive option
        exclusive = next((o for o in self.options if o.exclusive), None)
        if exclusive:
            return {exclusive: exclusive.inject(self)}

        return {option_cls: option_cls.inject(self) for option_cls in self.options}


predicates = ("default", "particular", "eager", "exclusive")


@attr.s
class Option:
    """An Option is a sequence of command line arguments bound to a starter mark."""

    marks: typing.Tuple[str, ...]
    nargs: typing.Union[int, bool]
    dependencies: typing.Dict[str, typing.Type["Option"]]
    default: typing.Any = missing
    eager: bool = False
    exclusive: bool = False
    serialize: typing.Callable

    value: typing.Any = attr.ib()

    def __init_subclass__(
        cls,
        *,
        marks=missing,
        help=missing,
        default=missing,
        particular=missing,
        eager=missing,
        exclusive=missing,
    ):
        args = locals()
        for name in ("marks", "help") + predicates:
            if (value := name in args and args[name]) is not missing:
                setattr(cls, name, value)

        if hasattr(cls, "serialize"):
            cls.nargs, cls.dependencies = interpret_nargs_dependencies(cls.serialize)
            if cls.eager and cls.dependencies:
                # XXX TODO allow other eager options as dependencies
                raise EagerDependencyConflict(
                    "An eager option must not have dependencies.",
                    cls.serialize,
                    cls.dependencies,
                )
        else:
            cls.nargs = 0
            cls.dependencies = {}

    @classmethod
    def option_marks(cls):
        for mark in cls.marks:
            yield (mark, cls)

    @classmethod
    def make(
        cls,
        *marks: str,
        help: str = None,
        default=missing,
        particular=single(),
        eager: bool = False,
        exclusive: bool = False,
    ) -> typing.Callable:
        def decorator(func) -> OptionType:
            name = func.__name__
            attributes = {
                "marks": marks,
                "serialize": classmethod(
                    particular(func) if callable(particular) else func
                ),
                "default": default,
                "eager": eager,
                "exclusive": exclusive,
                "__doc__": help or func.__doc__,
                "__repr__": lambda self: f"<{name}: {' '.join(marks)} => {repr(self.value)}>",
            }
            option_cls = type.__new__(type, name, (cls,), attributes)
            return option_cls

        return decorator

    @classmethod
    def consume(cls, *args: str) -> Match:
        match = Match()
        if not any(map(functools.partial(operator.eq, args[0]), cls.marks)):
            match.remaining = list(args)
        elif cls.nargs is True:
            match.add_option_args(cls, args)
        else:
            match.remaining = list(args[cls.nargs :])
            match.add_option_args(cls, args[0 : cls.nargs])
        return match

    @classmethod
    def inject(cls, parser: Parser):
        if cls in parser.values:
            return parser.values[cls]

        # every Option needs consumed arguments or a default
        if cls not in parser.option_args:
            if cls.default is missing:
                raise MissingOption(cls)

            value = parser.values[cls] = cls(
                cls.default(cls) if callable(cls.default) else cls.default
            )
            return value

        if not cls.dependencies:
            value = parser.values[cls] = cls(cls.serialize(*parser.option_args[cls]))
            return value

        dependencies = {
            name: dep.inject(parser) for name, dep in cls.dependencies.items()
        }

        value = parser.values[cls] = cls(
            cls.serialize(*parser.option_args[cls], **dependencies)
        )
        return value

    @classmethod
    def all_dependencies(cls):
        dependencies = set()
        for option_cls in cls.dependencies.values():
            dependencies.update(option_cls.all_dependencies())
            yield from option_cls.all_dependencies()
            for mark in option_cls.marks:
                yield mark, option_cls

    @classmethod
    def generate_help(cls):
        """Title, marks, predicates, help, deps.

        We need a basic scheme of layout:

        <---------- title ------------>

        <--------   <-- predicates --->
        ---------
        - marks -   <------------------
        ---------   -------------------
        -------->   -------------------
                    ------ help -------
                    -------------------
                    -------------------
                    ------------------>

                    <------------------
                    ------ deps -------
                    ------------------>

        So every section will be filled by the help implementation.
        The interface is an iterable per section.

        <mark1>     <args>
        <mark2>
        <markx>     <help>

                    <deps>


        <group>

                    <args>

                    <hekp>

                    <deps>

        <group> <subgroup>

                    <args>

                    <help>

                    <deps>
        """
        yield Sections(
            title=None,
            help=cls.__doc__,
            marks=list(cls.marks),
            predicates={predicate: getattr(cls, predicate) for predicate in predicates},
            deps=list(
                itertools.chain(
                    *map(operator.attrgetter("marks"), cls.dependencies.values())
                )
            ),
        )


@attr.s(auto_attribs=True)
class Sections:
    title: typing.Optional[str] = None
    help: typing.Optional[str] = None
    marks: typing.List[str] = attr.ib(factory=list)
    predicates: typing.Dict[str, typing.Any] = attr.ib(factory=dict)
    deps: typing.List[str] = attr.ib(factory=list)

    @property
    def marks_width(self):
        return max(map(len, self.marks))

    @property
    def title_width(self):
        return self.title and len(self.title) or 0


@attr.s
class Group(Option):
    members: typing.Dict[str, GroupType] = {}
    parent: typing.Optional[GroupType] = None

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.members = {}

    @classmethod
    def consume(cls, *args: str) -> Match:
        # the group has to consume sub members
        # otherwise we cannot ensure order
        member = args[0]
        remaining = args[1:]
        if member in cls.members:
            match = Match(remaining=list(remaining))
            match.add_option_args(cls, (member,))
            member_cls = cls.members[member]
            if remaining:
                member_match = member_cls.consume(*remaining)
                if not member_match.consumed:
                    match.add_option_args(member_cls, (None,))

                member_match.join(match)
                return member_match
            match.add_option_args(member_cls, (None,))
            return match
        return Match(remaining=list(args))

    @classmethod
    def _member_path(cls, match: Match):
        yield cls
        member_name = match.consumed[cls][0][0]
        if member_name:
            member_cls = cls.members[member_name]
            yield from member_cls._member_path(match)

    @classmethod
    def inject(cls, parser: Parser):
        if cls in parser.values:
            return parser.values[cls]
        path = list(cls._member_path(parser))
        value = None
        while path:
            member_cls = path.pop(0)
            value, parent_value = super(Group, member_cls).inject(parser), value
            value.parent = parent_value
        return value

    @classmethod
    def member(cls, name=None) -> typing.Callable:
        def decorator(func) -> GroupType:
            mark = name or func.__name__
            group_cls = cls.make(mark)(func)
            cls.members[mark], group_cls.parent = group_cls, cls
            cls.marks = cls.marks + (mark,)
            return group_cls

        return decorator
