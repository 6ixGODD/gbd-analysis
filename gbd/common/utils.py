from __future__ import annotations

import datetime
import pathlib
import re
import string
import sys
import typing

import typing_extensions


class ANSIFormatter:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    REVERSED = '\033[7m'

    @classmethod
    def format(cls, text: str, *styles: str) -> str:
        if cls.RESET in text:
            text = text.replace(cls.RESET, f"{cls.RESET}{''.join(styles)}")
        return f"{''.join(styles)}{text}{cls.RESET}"


class SafeFormatter(string.Formatter):
    @typing_extensions.override
    def format(self, __format_string: str, /, *args: typing.Any, **kwargs: typing.Any) -> str:
        result = ''
        try:
            for literal_text, field_name, format_spec, conversion in self.parse(__format_string):
                # Append the literal text
                result += literal_text

                # If there's a field, process it
                if field_name is not None:
                    try:
                        # Get the value
                        obj = self.get_value(field_name, args, kwargs)
                        # Convert and format the field
                        obj = self.convert_field(obj, conversion)
                        formatted = self.format_field(obj, format_spec or '')
                        result += formatted
                    except (KeyError, IndexError):
                        # Reconstruct the placeholder and leave it as is
                        placeholder = '{' + field_name
                        if conversion:
                            placeholder += '!' + conversion
                        if format_spec:
                            placeholder += ':' + format_spec
                        placeholder += '}'
                        result += placeholder
        except ValueError:
            result = __format_string
        return result

    @typing_extensions.override
    def get_value(
        self,
        key: typing.Union[int, str],
        args: typing.Sequence[typing.Any],
        kwargs: typing.Dict[str, typing.Any]
    ) -> typing.Any:
        if isinstance(key, int):
            if key < len(args):
                return args[key]
            else:
                raise IndexError(key)
        else:
            return kwargs[key]

    @typing_extensions.override
    def format_field(self, value: typing.Any, format_spec: str) -> str:
        try:
            return super().format_field(value, format_spec)
        except (KeyError, ValueError):
            return str(value)


class Logger:
    @staticmethod
    def _safe_format(msg: str, *args: typing.Any, **kwargs: typing.Any) -> str:
        return SafeFormatter().format(msg, *args, **kwargs)

    def _log(self, level: str, color: str, msg: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        timestamp = ANSIFormatter.format(str(datetime.datetime.now()), ANSIFormatter.GREEN)
        level_str = ANSIFormatter.format(f" |{level.ljust(8)}| ", color)
        formatted_msg = ANSIFormatter.format(self._safe_format(msg, *args, **kwargs), color, ANSIFormatter.BOLD)
        sys.stdout.write(f"{timestamp} {level_str} {formatted_msg}\n")
        sys.stdout.flush()

    def error(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._log("ERROR", ANSIFormatter.RED, msg, *args, **kwargs)

    def warning(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._log("WARNING", ANSIFormatter.YELLOW, msg, *args, **kwargs)

    def info(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._log("INFO", ANSIFormatter.BLUE, msg, *args, **kwargs)

    def debug(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        self._log("DEBUG", ANSIFormatter.CYAN, msg, *args, **kwargs)


def unique_path(root: pathlib.Path, name: str, /) -> pathlib.Path:
    if not (root / name).exists():
        (root / name).mkdir()
        return root / name
    i = 1
    while (path := root / f"{name}_{i}").exists():
        i += 1
    path.mkdir()
    return path


def to_snake_case(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\- ]", "", text).replace(" ", "_").replace("-", "_").lower()
