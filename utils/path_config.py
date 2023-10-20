
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import os


def get_paths():
    project = "/PATH/TO/maxpool_necessity/"

    logs = os.path.join(project, "logs")
    plots = os.path.join(project, "plots")
    results = os.path.join(project, "results")
    paths = {
        "results": results,
        "project": project,
        "plots": plots,
        "logs": logs
    }
    for k, v in paths.items():
        # print(k, v)
        os.makedirs(v, exist_ok=True)
        return paths
