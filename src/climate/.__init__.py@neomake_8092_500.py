import click


@click.group()
def group():
    print("group")


@click.command()
def cli():
    print("cli")


@group.command()
def group_cmd():
    print("group_cmd")


if __name__ == "__main__":
    group()
