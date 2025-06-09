from cookiecutter.main import cookiecutter
import click

@click.group()
def main():
    pass

@click.command()
def info():
    click.echo("Canvas: A flexible and modular deep learning project template.")

@click.command()
@click.argument('type', type=click.Choice(['sl', 'ssl', 'rl']))
def create(type):
    BASE_PATH = 'https://github.com/ramanakshay/canvas'
    DIRECTORY_PATH = f'templates/{type}'
    cookiecutter(BASE_PATH, directory=DIRECTORY_PATH)

main.add_command(info)
main.add_command(create)

if __name__ == '__main__':
    main()