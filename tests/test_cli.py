import abc
import collections
import inspect
import itertools
import sys
import typing

import attr
import multidict
import pytest


class CommandLine:
    def __init__(self, args=sys.argv):
        if isinstance(args, str):
            import shlex

            args = shlex.split(args)

        self.command = args[0]
        self.args = list(args[1:])
        self.pop = self.args.pop
        self.__getitem__ = self.args.__getitem__

    def take(self, i):
        try:
            return self.args[:i]
        finally:
            self.args = self.args[i:]


@attr.s(auto_attribs=True)
class Consumption:
    consumed: typing.List
    failed: typing.List
    remaining: typing.List


def consume(args, **consumers):
    """consume args to consumer."""
    consumed = multidict.MultiDict()
    available = list(args)
    stack = []
    stack.append(
        [consumer.state(key, available) for key, consumer in consumers.items()]
    )

    # store failed cnsumers per args state
    failed_state = collections.defaultdict(set)

    while available and stack:
        consumers = stack.pop(0)

        state = len(available)
        failed = set()
        while consumers:
            state = consumers.pop(0)
            try:
                taken, value = next(state.take)
            except StopIteration:
                # consumer is exhausted
                pass
            else:
                if not taken:
                    # this consumer failed
                    failed.add(state)
                else:
                    consumed.add(state.key, value)
                    # keep consumer, until exhausted
                    consumers.append(state)
                    # we push failed consumers and reset them
                    consumers.extend(failed)
                    failed.clear()
        else:
            if failed_state[state] == failed:
                # same set of consumers already tried
                break
            stack.append(list(failed))
            failed_state[state] = failed
    return Consumption(consumed, failed, available)


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
            if flag_arg in self.flags:
                value = value_arg
                args.pop(0)
                args.pop(0)
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
    stats = consume(cmd_line.args, config=Option("-c"), log_level=Option("--log-level"))

    stats = consume(
        stats.remaining,
        command=Arg(match=lambda x: x == "microservice"),
        verbose=Flag("-v"),
    )


def test_consume_wrong_order():
    cmd_line = CommandLine(
        "buvar --log-level debug -c config.toml microservice foo.bar"
    )

    stats = consume(cmd_line.args, config=Option("-c"), log_level=Option("--log-level"))
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


class Match:
    def __init__(self, context=None):
        # hold all consumer sets
        self.queue = []
        self.context = collections.ChainMap(context if context is not None else {})

    def me(self, **consumers):
        def decorator(func):
            self.queue.append((func, consumers))
            return func

        return decorator

    def cli(self, args=None):
        """A generator to consume a part of the arguments."""
        line = CommandLine(args if args else sys.argv)
        args = line.args
        __import__("pdb").set_trace()  # XXX BREAKPOINT
        while args and self.queue:
            func, consumers = self.queue.pop(0)
            stats = consume(args, **consumers)
            failed = stats.failed
            args = stats.remaining

            if failed:
                spec = inspect.getfullargspec(func)
                defaults = dict(
                    itertools.chain(
                        zip(reversed(spec.args or []), reversed(spec.defaults or [])),
                        (spec.kwonlydefaults or {}).items(),
                    )
                )

                def extract_defaults():
                    for state in failed:
                        key = state.key
                        if key in defaults:
                            stats.consumed[key] = defaults[key]
                        else:
                            yield state

                # try to find defaults
                failed = list(extract_defaults())

            if not failed:
                # generate args
                func_args = stats.consumed
                result = func(**dict(func_args))
                yield result

            else:
                __import__("pdb").set_trace()  # XXX BREAKPOINT


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
