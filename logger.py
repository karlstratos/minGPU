import os
import sys

from datetime import datetime


COLORS = {
    'purple': '\033[95m',
    'cyan': '\033[96m',
    'darkcyan': '\033[36m',
    'blue': '\033[94m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'bold': '\033[1m',
    'underline': '\033[4m',
    'end': '\033[0m'
}

def add_colors(string, colors):  # No support for nested colors
    for color in colors:
        string = COLORS.get(color, '') + string
    return string + COLORS['end']

def remove_colors(string):
    for value in COLORS.values():
        string = string.replace(value, '')
    return string


class Logger:

    def __init__(self, log_path=None, on=True, plain_file=True):
        self.log_path = log_path
        self.on = on
        self.plain_file = plain_file
        if self.on and self.log_path is not None:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def __call__(self, message, colors=[], stamp=True, newline=True,
                 force=False):
        message = add_colors(message, colors)

        if stamp:
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            message = f'{add_colors(time, ["green"])} | {message}'

        if self.on or force:
            sys.stdout.write(message)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()

            if self.on and self.log_path is not None:
                with open(self.log_path, 'a') as f:
                    f.write(remove_colors(message) if self.plain_file
                            else message)
                    if newline: f.write('\n')
                    f.flush()


if __name__ == '__main__':
    logger = Logger()
    logger('Plain text', stamp=False)
    logger('Stamped text')
    logger('Boldfaced', ['bold'])
    logger('Bold and red', ['bold', 'red'])

    mixed_colors = f'{add_colors("green", ["green"])}'
    mixed_colors += f' and {add_colors("blue", ["blue"])}'
    logger('Passing custom colored input: ' + mixed_colors)
    logger('Avoid the logger coloring when the input is already colored: '
           + mixed_colors, ['underline', 'cyan'])
