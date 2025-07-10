from cookiecutter.main import cookiecutter
import click
import os


@click.group()
def main():
    pass


@click.command()
def info():
    click.echo(
        "A modular, best-practice pytorch template for all your deep learning projects."
    )


@click.command()
@click.argument("type", type=click.Choice(["sl", "ssl", "rl"]))
def init(type):
    BASE_PATH = os.path.dirname(__file__) + "/templates"
    cookiecutter(BASE_PATH, directory=type)


main.add_command(info)
main.add_command(init)

if __name__ == "__main__":
    main()
