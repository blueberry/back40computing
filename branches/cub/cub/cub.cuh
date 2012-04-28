/******************************************************************************
 * 
 * Copyright (c) 2011-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/

/******************************************************************************
 * CUB umbrella include file
 ******************************************************************************/

#pragma once

#include <cta/cta_load.cuh>
#include <cta/cta_reduce.cuh>
#include <cta/cta_scan.cuh>
#include <cta/cta_store.cuh>

#include <thread/load.cuh>
#include <thread/reduce.cuh>
#include <thread/scan.cuh>
#include <thread/store.cuh>

#include <device_props.cuh>
#include <operators.cuh>
#include <debug.cuh>
#include <ptx_intrinsics.cuh>
#include <type_utils.cuh>
#include <work_distribution.cuh>

