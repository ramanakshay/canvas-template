import argparse
from cookiecutter.main import cookiecutter

def main():
    cookiecutter(BASE_PATH, directory=DIRECTORY_PATH)


parser = argparse.ArgumentParser()
parser.add_argument("command", help="enter command", default="create")
parser.add_argument("template", help="choose template", default="sl")
args = parser.parse_args()

BASE_PATH = 'https://github.com/ramanakshay/canvas'
DIRECTORY_PATH = f'templates/{args.template}'

if __name__ == '__main__':
    main()