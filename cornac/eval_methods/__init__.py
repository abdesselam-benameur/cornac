# Copyright 2018 The Cornac Authors. All Rights Reserved.
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
# ============================================================================

from .base_method import rating_eval
from .base_method import ranking_eval

from .base_method import BaseMethod
from .ratio_split import RatioSplit
from .stratified_split import StratifiedSplit
from .cross_validation import CrossValidation
from .next_basket_evaluation import NextBasketEvaluation
from .propensity_stratified_evaluation import PropensityStratifiedEvaluation

__all__ = ['BaseMethod',
           'RatioSplit',
           'StratifiedSplit',
           'CrossValidation',
           'NextBasketEvaluation',
           'PropensityStratifiedEvaluation']