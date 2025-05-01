# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import packaging.version

V2_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.

We introduced a new format since v2.0 which is not backward compatible with v1.x.
Please, use our conversion script. Modify the following command with your own task description:
```
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \\
    --repo-id {repo_id} \\
    --single-task "TASK DESCRIPTION."  # <---- /!\\ Replace TASK DESCRIPTION /!\\
```

A few examples to replace TASK DESCRIPTION: "Pick up the blue cube and place it into the bin.", "Insert the
peg into the socket.", "Slide open the ziploc bag.", "Take the elevator to the 1st floor.", "Open the top
cabinet, store the pot inside it then close the cabinet.", "Push the T-shaped block onto the T-shaped
target.", "Grab the spray paint on the shelf and place it in the bin on top of the robot dog.", "Fold the
sweatshirt.", ...

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

V21_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.
While current version of LeRobot is backward-compatible with it, the version of your dataset still uses global
stats instead of per-episode stats. Update your dataset stats to the new format using this command:
```
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py --repo-id={repo_id}
```

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

FUTURE_MESSAGE = """
The dataset you requested ({repo_id}) is only available in {version} format.
As we cannot ensure forward compatibility with it, please update your current version of lerobot.
"""

TROSSEN_V1_MESSAGE = """

The dataset you requested ({repo_id}) is in subversion {version} or lacks a defined subversion.

This subversion is incompatible with the current version of Interbotix/LeRobot.

Using incompatible subversions is discouraged, as joint actions may have larger-than-expected values in this code version.

We have introduced a new dataset format (Trossen Subversion 1.0) specifically for datasets involving Trossen robotic arms.
Key improvements in Trossen v1.0 include:
- Joint angles are now expressed in radians.
- Gripper values are represented in millimeters, with scaling removed.

To update your dataset to the new format, use the following conversion script:

```
python lerobot/scripts/convert_dataset_v21_to_v21_t10.py --repo-id {repo_id}
```

If you encounter a problem, contact Interbotix maintainers
or open an [issue on GitHub](https://github.com/Interbotix/lerobot/issues/new/choose).
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = V2_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


class SubVersionBackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = TROSSEN_V1_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)
