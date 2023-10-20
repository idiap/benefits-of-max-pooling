
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
from typing import Optional


def none_or_str(value: str) -> Optional[str]:
    if value == 'None':
        value = None
    return value


if __name__ == "__main__":
    # https://stackoverflow.com/questions/48295246/how-to-pass-none-keyword-as-command-line-argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_optional_str_arg", type=none_or_str, default="default")

    args = parser.parse_args()

    print("sys.argv = ")
    print(sys.argv)

    print(args.example_optional_str_arg)
