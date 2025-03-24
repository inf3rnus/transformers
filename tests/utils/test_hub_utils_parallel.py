# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import multiprocessing
import os
import shutil
import unittest

from diffusers import DiffusionPipeline

import transformers
from transformers.utils import (
    TRANSFORMERS_CACHE,
)


CACHE_DIR = os.path.join(
    TRANSFORMERS_CACHE, "models--hf-internal-testing--tiny-random-bert-sharded"
)


class GetFromCacheTestsParallel(unittest.TestCase):
    def setUp(self) -> None:
        # if os.path.exists(CACHE_DIR):
        #     shutil.rmtree(CACHE_DIR)

        os.environ["DISABLE_PARALLEL_LOADING"] = "false"
        os.environ["PARALLEL_LOADING_WORKERS"] = "2"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        multiprocessing.set_start_method("spawn", force=True)

    def tearDown(self) -> None:
        del os.environ["DISABLE_PARALLEL_LOADING"]
        del os.environ["HF_HUB_ENABLE_HF_TRANSFER"]

    def test_get_checkpoint_shard_files_integration(self):
        model = transformers.AutoModel.from_pretrained(
            "hf-internal-testing/tiny-random-bert-sharded",
        )
        self.assertIsNotNone(model)

        pipe = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", device_map="balanced"
        )
