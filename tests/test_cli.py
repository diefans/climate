from __future__ import annotations

import abc
import collections
import inspect
import itertools
import sys
import typing

import attr
import multidict
import pytest


class Argument:
    def __init__(self, token: str):
        self.token = token
        self.previuos = None
        self.next = None

    def __str__(self):
        return self.token

    def __repr__(self):
        return (
            f"-> {self.token}{' ' if self.next else ''}"
            f"{repr(self.next) if self.next else ''}"
        )

    def __len__(self):
        return 1 + (0 if self.next is None else len(self.next))

    def __iter__(self):
        yield self
        if self.next:
            yield from iter(self.next)

    def __getitem__(self, val):
        if isinstance(val, slice):

            def arguments():
                argument = self
                i = 0
                while argument:
                    if val.start is None or i <= val.start:
                        yield argument
                    i += 1
                    if val.stop and val.stop == i:
                        break
                    argument = argument.next

            return arguments()

        if val == 0:
            return self

        if self.next:
            return self.next[val - 1]

        raise IndexError("No more links.", self)

    def inject(self, item: Argument):
        item.previous, item.next = self, self.next
        if self.next:
            self.next.previous = item
        self.next = item
        return item

    def append(self, item: Argument):
        tail = self.tail
        item.previou, tail.next = tail, item
        return item

    @property
    def head(self):
        return self.previous or self

    @property
    def tail(self):
        return self.next or self

    @classmethod
    def chain(cls, *tokens):
        head, tail = None, None
        for token in tokens:
            argument = Argument(token)
            if tail is None:
                tail = head = argument
            else:
                tail = tail.inject(argument)
        return head


def test_argument():
    args = Argument.chain(*map(str, range(0, 5)))
    assert len(args) == 5
    assert ["0", "1", "2", "3", "4"] == list(map(str, args))


class CommandLine:
    def __init__(self, args=None):
        if isinstance(args, str):
            import shlex

            args = shlex.split(args)
        elif args is None:
            args = sys.argv

        self.command = args[0]
        self.arguments = Argument.chain(*args[1:])

    def consume(self, func):
        return func._match.apply(self.arguments)


class Mismatch(Exception):
    pass


class Consumer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def take(self, arguments):
        """Try to consume the next arguments.

        yield (None, None) if not matching
        yield (taken, value) if matching
        raise StopIteration if exhausted
        """

    def state(self, key, arguments):
        return ConsumerState(key, self, self.take(arguments))


@attr.s(auto_attribs=True, cmp=False)
class ConsumerState:
    key: str
    consumer: Consumer
    take: typing.Generator


class Arg(Consumer):
    def __init__(self, match=None):
        self.match = match

    def matches(self, arg):
        if isinstance(self.match, str) and self.match == arg:
            return True
        if isinstance(self.match, typing.Container) and arg in self.match:
            return True
        if callable(self.match) and self.match(arg):
            return True
        return False

    def take(self, args):
        """Consumes one argument."""
        if self.matches(args[0]):
            arg = args.pop(0)
            # parse arg into reasonable value
            value = arg
            yield [arg], value
        else:
            raise Mismatch("Argument mismatches", self, args)


class Option(Consumer):
    """An optional argument."""

    def __init__(self, *flags, multiple=False):
        self.flags = flags
        self.multiple = multiple

    def is_taken(self, taken):
        if not taken:
            return False

        if self.multiple is True:
            return False

        if taken < self.multiple:
            return False

        return True

    def take(self, args):
        taken = 0
        while args and not self.is_taken(taken):
            flag_arg, value_arg = args[:2]
            if flag_arg.token in self.flags:
                value = value_arg.token
                args = value_arg.next
                taken += 1
                yield [flag_arg, value_arg], value
            else:
                yield None, None


class Flag(Consumer):
    def __init__(self, *flags):
        self.flags = flags

    def take(self, args):
        taken = False
        while args and not taken:
            flag_arg = args[0]
            if flag_arg in self.flags:
                value = True
                args.pop(0)
                taken = True
                yield [flag_arg], value
            else:
                yield None, None


def test_consume_args():
    cmd_line = CommandLine(
        "buvar -c config.toml --log-level debug microservice foo.bar"
    )
    stats = Match().consume(
        cmd_line.args, config=Option("-c"), log_level=Option("--log-level")
    )

    stats = Match().consume(
        stats.remaining,
        command=Arg(match=lambda x: x == "microservice"),
        verbose=Flag("-v"),
    )


def test_consume_wrong_order():
    cmd_line = CommandLine(
        "buvar --log-level debug -c config.toml microservice foo.bar"
    )

    stats = Match().consume(
        cmd_line.args, config=Option("-c"), log_level=Option("--log-level")
    )
    assert stats.consumed == {"config": "config.toml", "log_level": "debug"}


def test_consume_find_dead_end():
    cmd_line = CommandLine(
        "buvar --log-level debug"
        " --log-level info"
        " -c config.toml"
        " microservice foo.bar"
    )

    stats = consume(
        cmd_line.args,
        foo=Option("--foo"),
        bar=Option("--bar"),
        log_level=Option("--log-level", multiple=2),
    )

    assert list(stats.consumed.items()) == [
        ("log_level", "debug"),
        ("log_level", "info"),
    ]
    assert stats.remaining == ["-c", "config.toml", "microservice", "foo.bar"]


def func_defaults(func):
    spec = inspect.getfullargspec(func)
    defaults = dict(
        itertools.chain(
            zip(reversed(spec.args or []), reversed(spec.defaults or [])),
            (spec.kwonlydefaults or {}).items(),
        )
    )
    return defaults


class Match:
    def __init__(self, **consumers):
        self.consumers = consumers
        self.func = None

    def __call__(self, func):
        self.func, func._match = func, self
        return func

    def consume(self, available):
        """Consume args until complete or with fail."""
        consumption = Consumption(self)
        stack = []
        stack.append(
            [consumer.state(key, available) for key, consumer in self.consumers.items()]
        )

        # store failed cnsumers per args state
        failed_state = collections.defaultdict(set)

        while available and stack:
            consumers = stack.pop(0)

            state = len(available)
            consumption.failed.clear()
            while consumers:
                consumer_state = consumers.pop(0)
                try:
                    taken, value = next(consumer_state.take)
                except StopIteration:
                    # consumer is exhausted
                    pass
                else:
                    if not taken:
                        # this consumer failed
                        consumption.failed.add(consumer_state)
                    else:
                        consumption.consumed.add(consumer_state.key, value)
                        # keep consumer, until exhausted
                        consumers.append(consumer_state)
                        # we push failed consumers and reset them to retry
                        consumers.extend(consumption.failed)
                        consumption.failed.clear()
            else:
                if consumption.failed and failed_state[state] == consumption.failed:
                    # same set of consumers already tried
                    # try to apply defaults
                    defaults = func_defaults(self.func)

                    def extract_defaults():
                        for state in consumption.failed:
                            key = state.key
                            if key in defaults:
                                consumption.consumed[key] = defaults[key]
                            else:
                                # give defaultless states back
                                yield state

                    # try to find defaults
                    consumption.failed = set(extract_defaults())
                    break
                stack.append(list(consumption.failed))
                failed_state[state] = consumption.failed

        return consumption

    def apply(self, args=None):
        """A generator to consume a part of the arguments."""
        consumption = self.consume(args)
        return consumption.apply()


@attr.s(auto_attribs=True)
class Consumption:
    match: Match
    consumed: multidict.MultiDict = multidict.MultiDict()
    failed: typing.Set = set()
    remaining: typing.List = []

    def apply(self):
        """Run the decorated code for this consumption."""
        if not self.failed:
            # generate args
            func_args = self.consumed
            result = self.match.func(**dict(func_args))
            return result
        return None


class Group:
    def __init__(self, name=None):
        self.name = name
        self.func = None

    def __call__(self, func):
        self.func = func
        return func


def test_match_consumer():
    line = CommandLine("command -c foo.toml microservice foo.bar")

    calls = {}

    @Match(config=Option("-c"), log_level=Option("--log-level"))
    def cli(config, log_level="debug"):
        calls[cli] = True
        assert (config, log_level) == ("foo.toml", "debug")
        # load config
        # include modules

        # @match.me(command=Arg("microservice"))
        # def microservice(command, modules=None):
        #     assert (command, modules) == ("microservice", None)

    results = line.consume(cli)
    assert calls == {cli: True}


def test_match():
    line = "command -c foo.toml microservice foo.bar"
    match = Match()

    @match.me(config=Option("-c"), log_level=Option("--log-level"))
    def cli(config, log_level="debug"):
        assert (config, log_level) == ("foo.toml", "debug")
        # load config
        # include modules

        @match.me(command=Arg("microservice"))
        def microservice(command, modules=None):
            assert (command, modules) == ("microservice", None)

    results = list(match.cli(line))
    assert results == [None, None]
