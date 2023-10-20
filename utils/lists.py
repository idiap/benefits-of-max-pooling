
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Any, List


def flatten_list_of_lists(ll: List[list]) -> list:
    flattened = [x for l in ll for x in l]
    return flattened


def splitter(to_split: list,
             split_elem: Any) -> list:
    # https://stackoverflow.com/questions/15357830/splitting-a-list-based-on-a-delimiter-word
    spl = [list(y) for x, y in itertools.groupby(to_split, lambda z: z == split_elem) if not x]
    return spl
