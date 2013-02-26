/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


/******************************************************************************
 * Tuning tool for establishing optimal consecutive removal granularity configuration types
 ******************************************************************************/

#include <stdio.h> 

#include <map>

#include <b40c/util/arch_dispatch.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/numeric_traits.cuh>
#include <b40c/util/parameter_generation.cuh>
#include <b40c/util/enactor_base.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/multiple_buffering.cuh>

#include <b40c/consecutive_reduction/problem_type.cuh>
#include <b40c/consecutive_reduction/policy.cuh>


// Test utils
#include "b40c_test_util.h"

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals, and utility types
 ******************************************************************************/

#ifndef TUNE_ARCH
	#define TUNE_ARCH (200)
#endif
#ifndef TUNE_SIZE
	#define TUNE_SIZE (4)
#endif

bool 	g_verbose;
int 	g_max_ctas = 0;
int 	g_iterations = 0;
bool 	g_verify;
int 	g_policy_id = 0;


struct KernelDetails
{
	int threads;
	int tile_elements;

	KernelDetails(
		int threads,
		int tile_elements) :
			threads(threads),
			tile_elements(tile_elements) {}
};


/******************************************************************************
 * Test wrappers for binary, associative operations
 ******************************************************************************/

template <typename T>
struct Sum
{
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return a + b;
	}
};

template <typename T>
struct Max
{
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}
};

template <typename T>
struct Equality
{
	// Equality test
	__host__ __device__ __forceinline__ bool operator()(const T &a, const T &b)
	{
		return a == b;
	}
};


/******************************************************************************
 * Utility routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage()
{
	printf("\ntune_consecutive_reduction [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-words>] [--verify]\n");
	printf("\n");
	printf("\t--v\tDisplays verbose configuration to the console.\n");
	printf("\n");
	printf("\t--verify\tChecks the result.\n");
	printf("\n");
	printf("\t--i\tPerforms the operation <num-iterations> times\n");
	printf("\t\t\ton the device. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of 32-bit words to comprise the sample problem\n");
	printf("\n");
	printf("\t--max-ctas\tThe number of CTAs to launch\n");
	printf("\n");
}


/******************************************************************************
 * Upsweep Tuning Parameter Enumerations and Ranges
 ******************************************************************************/

struct UpsweepTuning
{
	/**
	 * Tuning params
	 */
	enum Param
	{
		BEGIN,
			LOG_THREADS,
			LOG_LOAD_VEC_SIZE,
			LOG_LOADS_PER_TILE,
			LOG_SCHEDULE_GRANULARITY,
		END,
		CONSECUTIVE_SMEM_ASSIST,
	};

	/**
	 * Policy
	 */
	template <
		typename ProblemType,
		typename ParamList,
		typename BaseKernelPolicy =
			consecutive_reduction::upsweep::KernelPolicy <
				ProblemType,
				TUNE_ARCH,
				true,														// CHECK_ALIGNMENT
				0,															// MIN_CTA_OCCUPANCY,
				util::Access<ParamList, LOG_THREADS>::VALUE, 				// LOG_THREADS,
				util::Access<ParamList, LOG_LOAD_VEC_SIZE>::VALUE,			// LOG_LOAD_VEC_SIZE,
				util::Access<ParamList, LOG_LOADS_PER_TILE>::VALUE,			// LOG_LOADS_PER_TILE,
				B40C_LOG_WARP_THREADS(TUNE_ARCH),							// LOG_RAKING_THREADS,
				util::io::ld::NONE,											// READ_MODIFIER,
				util::io::st::NONE,											// WRITE_MODIFIER,
				util::Access<ParamList, LOG_SCHEDULE_GRANULARITY>::VALUE,	// LOG_SCHEDULE_GRANULARITY
				1> >														// CONSECUTIVE_SMEM_ASSIST

	struct KernelPolicy : BaseKernelPolicy
	{
		typedef typename ProblemType::KeyType 			KeyType;
		typedef typename ProblemType::ValueType			ValueType;
		typedef typename ProblemType::SizeT 			SizeT;
		typedef typename ProblemType::ReductionOp 		ReductionOp;
		typedef typename ProblemType::EqualityOp		EqualityOp;
		typedef typename ProblemType::SpineSizeT		SpineSizeT;

		typedef void (*KernelPtr)(KeyType*, ValueType*, ValueType*, SizeT*, ReductionOp, EqualityOp, util::CtaWorkDistribution<SizeT>);

		// Check if this configuration is worth compiling
		enum {
			REG_MULTIPLIER =
				((sizeof(KeyType) + 4 - 1) / 4) + 		// Keys
				((sizeof(ValueType) + 4 - 1) / 4) +		// Values
				((sizeof(SizeT) + 4 - 1) / 4),			// Ranks
			REGS_ESTIMATE = (REG_MULTIPLIER * KernelPolicy::TILE_ELEMENTS_PER_THREAD) + 2,
			EST_REGS_OCCUPANCY = B40C_SM_REGISTERS(TUNE_ARCH) / (REGS_ESTIMATE * KernelPolicy::THREADS),

			VALID_COMPILE =
				((BaseKernelPolicy::VALID > 0) &&
				(REGS_ESTIMATE < 63) &&
				((TUNE_ARCH >= 200) || (BaseKernelPolicy::READ_MODIFIER == util::io::ld::NONE)) &&
				((TUNE_ARCH >= 200) || (BaseKernelPolicy::WRITE_MODIFIER == util::io::st::NONE)) &&
				(BaseKernelPolicy::LOG_THREADS <= B40C_LOG_CTA_THREADS(TUNE_ARCH)) &&
				(EST_REGS_OCCUPANCY > 0)),
		};

		static std::string TypeString()
		{
			char buffer[32];
			sprintf(buffer, "%d, %d, %d",
				KernelPolicy::LOG_THREADS,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::LOG_LOADS_PER_TILE);
			return buffer;
		}

		template <int VALID, int DUMMY = 0>
		struct GenKernel
		{
			static KernelPtr Kernel() {
				return consecutive_reduction::upsweep::Kernel<KernelPolicy>;
			}
		};

		template <int DUMMY>
		struct GenKernel<0, DUMMY>
		{
			static KernelPtr Kernel() {
				return NULL;
			}
		};

		static KernelPtr Kernel() {
			return GenKernel<VALID_COMPILE>::Kernel();
		}
	};


	/**
	 * Ranges for the tuning params
	 */
	template <typename ParamList, int PARAM> struct Ranges;

	// LOG_THREADS
	template <typename ParamList>
	struct Ranges<ParamList, LOG_THREADS> {
		enum {
			MIN = 5,	// 32
			MAX = 10	// 1024
		};
	};

	// LOG_LOAD_VEC_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, LOG_LOAD_VEC_SIZE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// LOG_LOADS_PER_TILE
	template <typename ParamList>
	struct Ranges<ParamList, LOG_LOADS_PER_TILE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// CONSECUTIVE_SMEM_ASSIST
	template <typename ParamList>
	struct Ranges<ParamList, CONSECUTIVE_SMEM_ASSIST> {
		enum {
			MIN = 1,
			MAX = 1
		};
	};

	// LOG_SCHEDULE_GRANULARITY
	template <typename ParamList>
	struct Ranges<ParamList, LOG_SCHEDULE_GRANULARITY> {
		enum {
			MIN = util::Access<ParamList, LOG_THREADS>::VALUE +
				util::Access<ParamList, LOG_LOAD_VEC_SIZE>::VALUE +
				util::Access<ParamList, LOG_LOADS_PER_TILE>::VALUE,
			MAX = Ranges<ParamList, LOG_THREADS>::MAX +
				Ranges<ParamList, LOG_LOAD_VEC_SIZE>::MAX +
				Ranges<ParamList, LOG_LOADS_PER_TILE>::MAX
		};
	};
};


/******************************************************************************
 * Spine Tuning Parameter Enumerations and Ranges
 ******************************************************************************/

struct SpineTuning
{
	/**
	 * Tuning params
	 */
	enum Param
	{
		BEGIN,
			LOG_THREADS,
			LOG_LOAD_VEC_SIZE,
			LOG_LOADS_PER_TILE,
			LOG_SCHEDULE_GRANULARITY,
		END,
		CONSECUTIVE_SMEM_ASSIST,
	};

	/**
	 * Policy
	 */
	template <
		typename ProblemType,
		typename ParamList,
		typename BaseKernelPolicy =
			consecutive_reduction::upsweep::KernelPolicy <
				ProblemType,
				TUNE_ARCH,
				false,														// CHECK_ALIGNMENT
				1,															// MIN_CTA_OCCUPANCY,
				util::Access<ParamList, LOG_THREADS>::VALUE, 				// LOG_THREADS,
				util::Access<ParamList, LOG_LOAD_VEC_SIZE>::VALUE,			// LOG_LOAD_VEC_SIZE,
				util::Access<ParamList, LOG_LOADS_PER_TILE>::VALUE,			// LOG_LOADS_PER_TILE,
				B40C_LOG_WARP_THREADS(TUNE_ARCH),							// LOG_RAKING_THREADS,
				util::io::ld::NONE,											// READ_MODIFIER,
				util::io::st::NONE,											// WRITE_MODIFIER,
				util::Access<ParamList, LOG_SCHEDULE_GRANULARITY>::VALUE,	// LOG_SCHEDULE_GRANULARITY
				1> >														// CONSECUTIVE_SMEM_ASSIST

	struct KernelPolicy : BaseKernelPolicy
	{
		typedef typename ProblemType::KeyType 			KeyType;
		typedef typename ProblemType::ValueType			ValueType;
		typedef typename ProblemType::SizeT 			SizeT;
		typedef typename ProblemType::ReductionOp 		ReductionOp;
		typedef typename ProblemType::EqualityOp		EqualityOp;
		typedef typename ProblemType::SpineSizeT		SpineSizeT;

		typedef void (*KernelPtr)(ValueType*, ValueType*, SizeT*, SizeT*, SpineSizeT, ReductionOp);

		// Check if this configuration is worth compiling
		enum {
			REG_MULTIPLIER =
				((sizeof(ValueType) + 4 - 1) / 4) +		// Values
				((sizeof(SizeT) + 4 - 1) / 4),			// Ranks
			REGS_ESTIMATE = (REG_MULTIPLIER * KernelPolicy::TILE_ELEMENTS_PER_THREAD) + 2,
			EST_REGS_OCCUPANCY = B40C_SM_REGISTERS(TUNE_ARCH) / (REGS_ESTIMATE * KernelPolicy::THREADS),
/*
			// ptxas dies on this special case
			INVALID_SPECIAL =
				(TUNE_ARCH < 200) &&
				(sizeof(T) > 4) &&
				(BaseKernelPolicy::LOG_TILE_ELEMENTS > 9),
*/
			VALID_COMPILE =
				((BaseKernelPolicy::VALID > 0) &&
//				(INVALID_SPECIAL == 0) &&
				(REGS_ESTIMATE < 63) &&
				((TUNE_ARCH >= 200) || (BaseKernelPolicy::READ_MODIFIER == util::io::ld::NONE)) &&
				((TUNE_ARCH >= 200) || (BaseKernelPolicy::WRITE_MODIFIER == util::io::st::NONE)) &&
				(BaseKernelPolicy::LOG_THREADS <= B40C_LOG_CTA_THREADS(TUNE_ARCH)) &&
				(EST_REGS_OCCUPANCY > 0))
		};

		static std::string TypeString()
		{
			char buffer[32];
			sprintf(buffer, "%d, %d, %d",
				KernelPolicy::LOG_THREADS,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::LOG_LOADS_PER_TILE);
			return buffer;
		}

		template <int VALID, int DUMMY = 0>
		struct GenKernel
		{
			static KernelPtr Kernel() {
				return consecutive_reduction::spine::Kernel<KernelPolicy>;
			}
		};

		template <int DUMMY>
		struct GenKernel<0, DUMMY>
		{
			static KernelPtr Kernel() {
				return NULL;
			}
		};

		static KernelPtr Kernel() {
			return GenKernel<VALID_COMPILE>::Kernel();
		}
	};


	/**
	 * Ranges for the tuning params
	 */
	template <typename ParamList, int PARAM> struct Ranges;

	// LOG_THREADS
	template <typename ParamList>
	struct Ranges<ParamList, LOG_THREADS> {
		enum {
			MIN = 5,	// 32
			MAX = 10	// 1024
		};
	};

	// LOG_LOAD_VEC_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, LOG_LOAD_VEC_SIZE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// LOG_LOADS_PER_TILE
	template <typename ParamList>
	struct Ranges<ParamList, LOG_LOADS_PER_TILE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// CONSECUTIVE_SMEM_ASSIST
	template <typename ParamList>
	struct Ranges<ParamList, CONSECUTIVE_SMEM_ASSIST> {
		enum {
			MIN = 1,
			MAX = 1
		};
	};

	// LOG_SCHEDULE_GRANULARITY
	template <typename ParamList>
	struct Ranges<ParamList, LOG_SCHEDULE_GRANULARITY> {
		enum {
			MIN = util::Access<ParamList, LOG_THREADS>::VALUE +
				util::Access<ParamList, LOG_LOAD_VEC_SIZE>::VALUE +
				util::Access<ParamList, LOG_LOADS_PER_TILE>::VALUE,
			MAX = MIN
		};
	};

};


/******************************************************************************
 * Downsweep Tuning Parameter Enumerations and Ranges
 ******************************************************************************/

struct DownsweepTuning
{
	/**
	 * Tuning params
	 */
	enum Param
	{
		BEGIN,
			LOG_THREADS,
			LOG_LOAD_VEC_SIZE,
			LOG_LOADS_PER_TILE,
			LOG_SCHEDULE_GRANULARITY,
		END,
		CONSECUTIVE_SMEM_ASSIST,
	};

	/**
	 * Policy
	 */
	template <
		typename ProblemType,
		typename ParamList,
		typename BaseKernelPolicy =
			consecutive_reduction::downsweep::KernelPolicy <
				ProblemType,
				TUNE_ARCH,
				true,														// CHECK_ALIGNMENT
				0,															// MIN_CTA_OCCUPANCY,
				util::Access<ParamList, LOG_THREADS>::VALUE, 				// LOG_THREADS,
				util::Access<ParamList, LOG_LOAD_VEC_SIZE>::VALUE,			// LOG_LOAD_VEC_SIZE,
				util::Access<ParamList, LOG_LOADS_PER_TILE>::VALUE,			// LOG_LOADS_PER_TILE,
				B40C_LOG_WARP_THREADS(TUNE_ARCH),							// LOG_RAKING_THREADS,
				util::io::ld::NONE,											// READ_MODIFIER,
				util::io::st::NONE,											// WRITE_MODIFIER,
				util::Access<ParamList, LOG_SCHEDULE_GRANULARITY>::VALUE,	// LOG_SCHEDULE_GRANULARITY
				false,														// TWO_PHASE_SCATTER
				1> >														// CONSECUTIVE_SMEM_ASSIST

	struct KernelPolicy : BaseKernelPolicy
	{
		typedef typename ProblemType::KeyType 			KeyType;
		typedef typename ProblemType::ValueType			ValueType;
		typedef typename ProblemType::SizeT 			SizeT;
		typedef typename ProblemType::ReductionOp 		ReductionOp;
		typedef typename ProblemType::EqualityOp		EqualityOp;
		typedef typename ProblemType::SpineSizeT		SpineSizeT;

		typedef void (*KernelPtr)(KeyType*, KeyType*, ValueType*, ValueType*, ValueType*,  SizeT*, SizeT*, ReductionOp, EqualityOp, util::CtaWorkDistribution<SizeT>);

		// Check if this configuration is worth compiling
		enum {
			REG_MULTIPLIER =
				((sizeof(KeyType) + 4 - 1) / 4) + 		// keys
				((sizeof(ValueType) + 4 - 1) / 4) +		// values
				((sizeof(int) + 4 - 1) / 4) +			// head_flags
				((sizeof(SizeT) + 4 - 1) / 4),			// ranks
			REGS_ESTIMATE = (REG_MULTIPLIER * KernelPolicy::TILE_ELEMENTS_PER_THREAD) + 6,

			EST_REGS_OCCUPANCY = B40C_SM_REGISTERS(TUNE_ARCH) / (REGS_ESTIMATE * KernelPolicy::THREADS),

			VALID_COMPILE =
				((BaseKernelPolicy::VALID > 0) &&
				(REGS_ESTIMATE < 63) &&
				((TUNE_ARCH >= 200) || (BaseKernelPolicy::READ_MODIFIER == util::io::ld::NONE)) &&
				((TUNE_ARCH >= 200) || (BaseKernelPolicy::WRITE_MODIFIER == util::io::st::NONE)) &&
				(BaseKernelPolicy::LOG_THREADS <= B40C_LOG_CTA_THREADS(TUNE_ARCH)) &&
				(EST_REGS_OCCUPANCY > 0)),
		};

		static std::string TypeString()
		{
			char buffer[32];
			sprintf(buffer, "%d, %d, %d",
				KernelPolicy::LOG_THREADS,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::LOG_LOADS_PER_TILE);
			return buffer;
		}

		template <int VALID, int DUMMY = 0>
		struct GenKernel
		{
			static KernelPtr Kernel() {
				return consecutive_reduction::downsweep::Kernel<KernelPolicy>;
			}
		};

		template <int DUMMY>
		struct GenKernel<0, DUMMY>
		{
			static KernelPtr Kernel() {
				return NULL;
			}
		};

		static KernelPtr Kernel() {
			return GenKernel<VALID_COMPILE>::Kernel();
		}
	};


	/**
	 * Ranges for the tuning params
	 */
	template <typename ParamList, int PARAM> struct Ranges;

	// LOG_THREADS
	template <typename ParamList>
	struct Ranges<ParamList, LOG_THREADS> {
		enum {
			MIN = 5,	// 32
			MAX = 10	// 1024
		};
	};

	// LOG_LOAD_VEC_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, LOG_LOAD_VEC_SIZE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// LOG_LOADS_PER_TILE
	template <typename ParamList>
	struct Ranges<ParamList, LOG_LOADS_PER_TILE> {
		enum {
			MIN = 0,
			MAX = 2
		};
	};

	// CONSECUTIVE_SMEM_ASSIST
	template <typename ParamList>
	struct Ranges<ParamList, CONSECUTIVE_SMEM_ASSIST> {
		enum {
			MIN = 1,
			MAX = 1
		};
	};

	// LOG_SCHEDULE_GRANULARITY
	template <typename ParamList>
	struct Ranges<ParamList, LOG_SCHEDULE_GRANULARITY> {
		enum {
			MIN = util::Access<ParamList, LOG_THREADS>::VALUE +
				util::Access<ParamList, LOG_LOAD_VEC_SIZE>::VALUE +
				util::Access<ParamList, LOG_LOADS_PER_TILE>::VALUE,
			MAX = Ranges<ParamList, LOG_THREADS>::MAX +
				Ranges<ParamList, LOG_LOAD_VEC_SIZE>::MAX +
				Ranges<ParamList, LOG_LOADS_PER_TILE>::MAX
		};
	};
};


/******************************************************************************
 * General Tuning Parameter Enumerations and Ranges
 ******************************************************************************/

struct GeneralTuning
{
	enum Param
	{
		PARAM_BEGIN,
		PARAM_END,

		// Parameters below here are currently not part of the tuning sweep
		READ_MODIFIER,
		WRITE_MODIFIER,
		UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE,
		LOG_SCHEDULE_GRANULARITY,
	};


	/**
	 * Ranges for the tuning params
	 */
	template <typename ParamList, int PARAM> struct Ranges;

	// READ_MODIFIER
	template <typename ParamList>
	struct Ranges<ParamList, READ_MODIFIER> {
		enum {
			MIN = util::io::ld::NONE,
			MAX = util::io::ld::LIMIT - 1,
		};
	};

	// WRITE_MODIFIER
	template <typename ParamList>
	struct Ranges<ParamList, WRITE_MODIFIER> {
		enum {
			MIN = util::io::st::NONE,
			MAX = util::io::st::LIMIT - 1,
		};
	};

	// UNIFORM_SMEM_ALLOCATION
	template <typename ParamList>
	struct Ranges<ParamList, UNIFORM_SMEM_ALLOCATION> {
		enum {
			MIN = 0,
			MAX = 1
		};
	};

	// UNIFORM_GRID_SIZE
	template <typename ParamList>
	struct Ranges<ParamList, UNIFORM_GRID_SIZE> {
		enum {
			MIN = 0,
			MAX = 1
		};
	};
};


/******************************************************************************
 * Generators
 ******************************************************************************/



/**
 * Tuple callback generator
 */
template <
	typename ProblemType,
	typename Tuning,
	typename ConfigMap>
struct Callback
{
	typedef typename ConfigMap::mapped_type 	GrainMap;				// int -> LaunchDetails
	typedef typename ConfigMap::value_type 		ConfigMapPair;			// (string, GrainMap)
	typedef typename GrainMap::mapped_type 		LaunchDetails;			// (KernelDetails, kernel function ptr)
	typedef typename GrainMap::value_type 		GrainLaunchDetails;		// (int, LaunchDetails)


	ConfigMap *config_map;

	Callback(ConfigMap *config_map) : config_map(config_map) {}

	void Generate()
	{
		util::ParamListSweep<
			Tuning::BEGIN + 1,
			Tuning::END,
			Tuning::template Ranges>::template Invoke<util::EmptyTuple>(*this);
	}

	template <typename ParamList>
	void Invoke()
	{
		typedef typename Tuning::template KernelPolicy<
			ProblemType,
			ParamList> KernelPolicy;

		// Type string for this config family
		std::string typestring = KernelPolicy::TypeString();

		// Create pairing between kernel-details and kernel-pointer
		LaunchDetails launch_details(
			KernelDetails(KernelPolicy::THREADS, KernelPolicy::TILE_ELEMENTS),
			KernelPolicy::Kernel());

		// Create pairing between granularity and launch-details
		GrainLaunchDetails grain_launch_details(
			KernelPolicy::LOG_SCHEDULE_GRANULARITY,
			launch_details);

		// Check to see if we've started a grain list
		if (config_map->find(typestring) == config_map->end()) {

			// Not found.  Insert grain pair into new grain map, insert grain map into config map
			GrainMap grain_map;
			grain_map.insert(grain_launch_details);

			config_map->insert(ConfigMapPair(typestring, grain_map));

		} else {

			// Add this scheduling granularity to the config list
			config_map->find(typestring)->second.insert(grain_launch_details);
		}
	}
};



template <typename ProblemType>
struct Enactor : public util::EnactorBase
{
	typedef typename ProblemType::KeyType 			KeyType;
	typedef typename ProblemType::ValueType			ValueType;
	typedef typename ProblemType::SizeT 			SizeT;
	typedef typename ProblemType::ReductionOp 		ReductionOp;
	typedef typename ProblemType::EqualityOp		EqualityOp;
	typedef typename ProblemType::SpineSizeT		SpineSizeT;

	// Kernel pointer types
	typedef void (*UpsweepKernelPtr)(KeyType*, ValueType*, ValueType*, SizeT*, ReductionOp, EqualityOp, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelPtr)(ValueType*, ValueType*, SizeT*, SizeT*, SpineSizeT, ReductionOp);
	typedef void (*DownsweepKernelPtr)(KeyType*, KeyType*, ValueType*, ValueType*, ValueType*,  SizeT*, SizeT*, ReductionOp, EqualityOp, util::CtaWorkDistribution<SizeT>);

	typedef std::pair<KernelDetails, UpsweepKernelPtr> 		UpsweepLaunchDetails;
	typedef std::pair<KernelDetails, SpineKernelPtr> 		SpineLaunchDetails;
	typedef std::pair<KernelDetails, DownsweepKernelPtr> 	DownsweepLaunchDetails;

	// Config grain-map types (LOG_GRANULARITY -> kernel pointer)
	typedef std::map<int, UpsweepLaunchDetails> 		UpsweepGrainMap;
	typedef std::map<int, SpineLaunchDetails> 			SpineGrainMap;
	typedef std::map<int, DownsweepLaunchDetails>		DownsweepGrainMap;

	// Config map types (tune-string -> grain map)
	typedef std::map<std::string, UpsweepGrainMap>		UpsweepMap;
	typedef std::map<std::string, SpineGrainMap> 		SpineMap;
	typedef std::map<std::string, DownsweepGrainMap>	DownsweepMap;

	// Configuration maps
	UpsweepMap 		upsweep_configs;
	SpineMap 		spine_configs;
	DownsweepMap 	downsweep_configs;

	// Temporary device storage needed for scanning value partials produced
	// by separate CTAs
	util::Spine partial_spine;

	// Temporary device storage needed for scanning flag partials produced
	// by separate CTAs
	util::Spine flag_spine;

	util::DoubleBuffer<KeyType, ValueType> 	d_problem_storage;
	SizeT 										num_elements;
	ReductionOp 								reduction_op;
	EqualityOp									equality_op;
	SizeT										*d_num_compacted;

	util::DoubleBuffer<KeyType, ValueType> 	h_problem_storage;	// host problem storage (selector points to input, but output contains reference result)
	SizeT 										h_num_compacted;		// number of elements in reference result

	/**
	 * Constructor
	 */
	Enactor(
		ReductionOp reduction_op,
		EqualityOp equality_op) :
			reduction_op(reduction_op),
			equality_op(equality_op),
			d_num_compacted(NULL)
	{
		// Pre-allocate our spines

		if (partial_spine.Setup<long long>(SmCount() * 8 * 8)) exit(1);
		if (flag_spine.Setup<long long>(SmCount() * 8 * 8)) exit(1);

		// Generate all config maps

		Callback<ProblemType, UpsweepTuning, UpsweepMap> 		upsweep_callback(&upsweep_configs);
		Callback<ProblemType, SpineTuning, SpineMap> 			spine_callback(&spine_configs);
		Callback<ProblemType, DownsweepTuning, DownsweepMap> 	downsweep_callback(&downsweep_configs);

		upsweep_callback.Generate();
		spine_callback.Generate();
		downsweep_callback.Generate();
	}


	/**
	 *
	 */
	cudaError_t RunSample(
		int log_schedule_granularity,
		UpsweepLaunchDetails upsweep_details,
		SpineLaunchDetails spine_details,
		DownsweepLaunchDetails downsweep_details)
	{
		const bool OVERSUBSCRIBED_GRID_SIZE = true;
		const bool UNIFORM_SMEM_ALLOCATION = false;
		const bool UNIFORM_GRID_SIZE = false;

		cudaError_t retval = cudaSuccess;
		do {

			// Max CTA occupancy for the actual target device
			int max_cta_occupancy;
			if (retval = MaxCtaOccupancy(
				max_cta_occupancy,
				upsweep_details.second,
				upsweep_details.first.threads,
				downsweep_details.second,
				downsweep_details.first.threads)) break;

			// Compute sweep grid size
			int sweep_grid_size = GridSize(
				OVERSUBSCRIBED_GRID_SIZE,
				1 << log_schedule_granularity,
				max_cta_occupancy,
				num_elements,
				g_max_ctas);

			// Compute spine elements: one element per CTA, rounded
			// up to nearest spine tile size
			int spine_elements = ((sweep_grid_size + spine_details.first.tile_elements - 1) / spine_details.first.tile_elements) *
				spine_details.first.tile_elements;

			// Obtain a CTA work distribution
			util::CtaWorkDistribution<SizeT> work;
			work.Init(num_elements, sweep_grid_size, log_schedule_granularity);

			if (ENACTOR_DEBUG) {
				printf("Work: ");
				work.Print();
			}

			// Make sure our spine is big enough
			if (retval = partial_spine.Setup<ValueType>(spine_elements)) break;
			if (retval = flag_spine.Setup<SizeT>(spine_elements)) break;

			// If we're to output the compacted sizes to device memory, write out
			// compacted size to the last element of our flag spine instead
			d_num_compacted = ((SizeT*) flag_spine()) + spine_elements - 1;

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option: make sure all kernels have the same overall smem allocation
			if (UNIFORM_SMEM_ALLOCATION) if (retval = PadUniformSmem(
				dynamic_smem,
				upsweep_details.second,
				spine_details.second,
				downsweep_details.second)) break;

			// Tuning option: make sure that all kernels launch the same number of CTAs)
			if (UNIFORM_GRID_SIZE) grid_size[1] = grid_size[0];

			if (ENACTOR_DEBUG) {
				printf("Upsweep<<<%d,%d,%d>>> Spine<<<%d,%d,%d>>> Downsweep<<<%d,%d,%d>>>\n",
					grid_size[0], upsweep_details.first.threads, dynamic_smem[0],
					grid_size[1], spine_details.first.threads, dynamic_smem[1],
					grid_size[2], downsweep_details.first.threads, dynamic_smem[2]);
			}

			// Upsweep scan into spine
			upsweep_details.second<<<grid_size[0], upsweep_details.first.threads, dynamic_smem[0]>>>(
				d_problem_storage.d_keys[d_problem_storage.selector],
				d_problem_storage.d_values[d_problem_storage.selector],
				(ValueType*) partial_spine(),
				(SizeT*) flag_spine(),
				reduction_op,
				equality_op,
				work);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Spine scan
			spine_details.second<<<grid_size[1], spine_details.first.threads, dynamic_smem[1]>>>(
				(ValueType*) partial_spine(),
				(ValueType*) partial_spine(),
				(SizeT*) flag_spine(),
				(SizeT*) flag_spine(),
				spine_elements,
				reduction_op);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

			// Downsweep from spine
			downsweep_details.second<<<grid_size[2], downsweep_details.first.threads, dynamic_smem[2]>>>(
				d_problem_storage.d_keys[d_problem_storage.selector],
				d_problem_storage.d_keys[d_problem_storage.selector ^ 1],
				d_problem_storage.d_values[d_problem_storage.selector],
				d_problem_storage.d_values[d_problem_storage.selector ^ 1],
				(ValueType*) partial_spine(),
				(SizeT*) flag_spine(),
				d_num_compacted,
				reduction_op,
				equality_op,
				work);

			if (ENACTOR_DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor DownsweepKernel failed ", __FILE__, __LINE__, ENACTOR_DEBUG))) break;

		} while (0);

		return retval;
	}


	/**
	 *
	 */
	void TimeSample(
		int log_schedule_granularity,
		UpsweepLaunchDetails upsweep_details,
		SpineLaunchDetails spine_details,
		DownsweepLaunchDetails downsweep_details)
	{
		// Check if valid for dispatch
		if (!upsweep_details.second || !spine_details.second || !downsweep_details.second) {
			return;
		}

		// Invoke kernels (warmup)
		ENACTOR_DEBUG = g_verbose;
		if (RunSample(
			log_schedule_granularity,
			upsweep_details,
			spine_details,
			downsweep_details))
		{
			exit(1);
		}
		ENACTOR_DEBUG = false;

		// Perform the timed number of iterations
		GpuTimer timer;
		double elapsed = 0;
		for (int i = 0; i < g_iterations; i++) {

			// Start cuda timing record
			timer.Start();

			// Invoke kernels
			if (RunSample(
				log_schedule_granularity,
				upsweep_details,
				spine_details,
				downsweep_details))
			{
				exit(1);
			}

			// End cuda timing record
			timer.Stop();
			elapsed += timer.ElapsedMillis();

			// Flushes any stdio from the GPU
			if (util::B40CPerror(cudaThreadSynchronize(), "TimedCopy cudaThreadSynchronize failed: ", __FILE__, __LINE__)) {
				exit(1);
			}
		}

		// Display timing information
		long long bytes = ((num_elements * 2) + h_num_compacted) * (sizeof(KeyType) + sizeof(ValueType));
		double avg_runtime = elapsed / g_iterations;
		double throughput =  0.0;
		double bandwidth =  0.0;

		if (avg_runtime > 0.0) {
			throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
			bandwidth = bytes / avg_runtime / 1000.0 / 1000.0;
		}

		printf(", %f, %f, %f, ",
			avg_runtime, throughput, bandwidth);
		fflush(stdout);

		if (g_verify) {

			CompareDeviceResults(
				h_problem_storage.d_keys[1],
				d_problem_storage.d_keys[1],
				h_num_compacted);
			printf(", ");
			CompareDeviceResults(
				h_problem_storage.d_values[1],
				d_problem_storage.d_values[1],
				h_num_compacted);
			printf(", ");
			CompareDeviceResults(
				&h_num_compacted,
				d_num_compacted,
				1);
		}
	}


	/**
	 * Iterates over configuration space
	 */
	void IterateConfigSpace()
	{
		int config_id = 0;

		// Iterate upsweep configs
		for (typename UpsweepMap::iterator upsweep_config_itr = upsweep_configs.begin();
			upsweep_config_itr != upsweep_configs.end();
			upsweep_config_itr++)
		{
			std::string upsweep_string = upsweep_config_itr->first;

			// Iterate downsweep configs
			for (typename DownsweepMap::iterator downsweep_config_itr = downsweep_configs.begin();
				downsweep_config_itr != downsweep_configs.end();
				downsweep_config_itr++)
			{
				std::string downsweep_string = downsweep_config_itr->first;

				typename UpsweepGrainMap::iterator upsweep_grain_itr = upsweep_config_itr->second.begin();
				typename DownsweepGrainMap::iterator downsweep_grain_itr = downsweep_config_itr->second.begin();

				while (true) {

					if ((upsweep_grain_itr == upsweep_config_itr->second.end()) ||
						(downsweep_grain_itr == downsweep_config_itr->second.end()))
					{
						// Could not match grain

						printf("Could not match upsweep(%s) with downsweep(%s)\n",
							upsweep_string.c_str(),
							downsweep_string.c_str());

						exit(1);

					}
					else if (upsweep_grain_itr->first == downsweep_grain_itr->first)
					{
						// Matched grain

						// Iterate spine configs
						for (typename SpineMap::iterator spine_config_itr = spine_configs.begin();
							spine_config_itr != spine_configs.end();
							spine_config_itr++)
						{
							std::string spine_string = spine_config_itr->first;

							printf("%d, %d, %s, %s, %s",
								config_id,
								upsweep_grain_itr->first,
								upsweep_string.c_str(),
								spine_string.c_str(),
								downsweep_string.c_str());
							config_id++;

							TimeSample(
								upsweep_grain_itr->first,
								upsweep_grain_itr->second,
								spine_config_itr->second.begin()->second,
								downsweep_grain_itr->second);

							printf("\n");
							fflush(stdout);
						}

						break;

					} else if (upsweep_grain_itr->first < downsweep_grain_itr->first) {
						upsweep_grain_itr++;
					} else {
						downsweep_grain_itr++;
					}
				}
			}
		}
	}


	/**
	 * Creates an example problem and then dispatches the iterations
	 * to the GPU for the given number of iterations, displaying runtime information.
	 */
	void Test(SizeT num_elements)
	{
		this->num_elements = num_elements;

	    // Allocate the consecutive reduction problem on the host
		h_problem_storage.d_keys[0] = (KeyType*) malloc(num_elements * sizeof(KeyType));
		h_problem_storage.d_keys[1] = (KeyType*) malloc(num_elements * sizeof(KeyType));
		h_problem_storage.d_values[0] = (ValueType*) malloc(num_elements * sizeof(ValueType));
		h_problem_storage.d_values[1] = (ValueType*) malloc(num_elements * sizeof(ValueType));

		if (!h_problem_storage.d_keys[0] || !h_problem_storage.d_keys[1] || !h_problem_storage.d_values[0] || !h_problem_storage.d_values[1]){
			fprintf(stderr, "Host malloc of problem data failed\n");
			exit(1);
		}

		// Initialize problem
		if (g_verbose) printf("Input problem: \n");
		for (int i = 0; i < num_elements; i++) {
			h_problem_storage.d_keys[0][i] = (i / 7) & 1;							// Toggle every 7 elements
	//		util::RandomBits<T>(h_problem_storage.d_keys[0][i], 1, 1);				// Entropy-reduced random 0|1 values: roughly 26 / 64 elements toggled

			h_problem_storage.d_values[0][i] = 1;

/*
			if (g_verbose) {
				printf("(%lld, %lld), ",
					(long long) h_problem_storage.d_keys[0][i],
					(long long) h_problem_storage.d_values[0][i]);
			}
*/
		}
//		if (g_verbose) printf("\n");


		// Compute reference solution
		h_num_compacted = 0;
		h_problem_storage.d_keys[1][0] = h_problem_storage.d_keys[0][0];

		for (SizeT i = 0; i < num_elements; ++i) {

			if (h_problem_storage.d_keys[1][h_num_compacted] != h_problem_storage.d_keys[0][i]) {

				h_num_compacted++;
				h_problem_storage.d_keys[1][h_num_compacted] = h_problem_storage.d_keys[0][i];
				h_problem_storage.d_values[1][h_num_compacted] = h_problem_storage.d_values[0][i];

			} else {

				if (i == 0) {
					h_problem_storage.d_values[1][h_num_compacted] =
						h_problem_storage.d_values[0][i];
				} else {
					h_problem_storage.d_values[1][h_num_compacted] = reduction_op(
						h_problem_storage.d_values[1][h_num_compacted],
						h_problem_storage.d_values[0][i]);
				}
			}
		}
		h_num_compacted++;

		// Allocate device storage
		if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_keys[0], sizeof(KeyType) * num_elements),
			"TimedConsecutiveReduction cudaMalloc d_keys failed: ", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_keys[1], sizeof(KeyType) * num_elements),
			"TimedConsecutiveReduction cudaMalloc d_keys failed: ", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_values[0], sizeof(ValueType) * num_elements),
			"TimedConsecutiveReduction cudaMalloc d_values failed: ", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaMalloc((void**) &d_problem_storage.d_values[1], sizeof(ValueType) * num_elements),
			"TimedConsecutiveReduction cudaMalloc d_values failed: ", __FILE__, __LINE__)) exit(1);

		// Move a fresh copy of the problem into device storage
		if (util::B40CPerror(cudaMemcpy(
				d_problem_storage.d_keys[0],
				h_problem_storage.d_keys[0],
				sizeof(KeyType) * num_elements,
				cudaMemcpyHostToDevice),
			"TimedConsecutiveReduction cudaMemcpy d_keys failed: ", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaMemcpy(
				d_problem_storage.d_values[0],
				h_problem_storage.d_values[0],
				sizeof(ValueType) * num_elements,
				cudaMemcpyHostToDevice),
			"TimedConsecutiveReduction cudaMemcpy d_values failed: ", __FILE__, __LINE__)) exit(1);

		// Iterate configuration space
		IterateConfigSpace();

		// Free allocated memory
	    if (d_problem_storage.d_keys[0]) cudaFree(d_problem_storage.d_keys[0]);
	    if (d_problem_storage.d_keys[1]) cudaFree(d_problem_storage.d_keys[1]);
	    if (d_problem_storage.d_values[0]) cudaFree(d_problem_storage.d_values[0]);
	    if (d_problem_storage.d_values[1]) cudaFree(d_problem_storage.d_values[1]);

		// Free our allocated host memory
		if (h_problem_storage.d_keys[0]) free(h_problem_storage.d_keys[0]);
		if (h_problem_storage.d_keys[1]) free(h_problem_storage.d_keys[1]);
		if (h_problem_storage.d_values[0]) free(h_problem_storage.d_values[0]);
		if (h_problem_storage.d_values[1]) free(h_problem_storage.d_values[1]);
	}


};



/******************************************************************************
 * Test
 ******************************************************************************/



/**
 * Creates an example problem and then dispatches the iterations
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<
	typename KeyType,
	typename ValueType,
	typename SizeT,
	typename ReductionOp,
	typename EqualityOp>
void Test(
	SizeT num_elements,
	ReductionOp reduction_op,
	EqualityOp equality_op)
{
	typedef consecutive_reduction::ProblemType<
		KeyType,
		ValueType,
		SizeT,
		ReductionOp,
		EqualityOp> ProblemType;

	// Create enactor
	Enactor<ProblemType> enactor(
		reduction_op,
		equality_op);

	// Run test
	enactor.Test(num_elements);
}




/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{

	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	// Seed random number generator
	srand(0);				// presently deterministic

	// Use 32-bit integer for array indexing
	typedef int SizeT;
	SizeT num_elements = 1024;

	// Parse command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
    g_verify = args.CheckCmdLineFlag("verify");
	g_verbose = args.CheckCmdLineFlag("v");

	util::CudaProperties cuda_props;

	printf("Test Scan: %d iterations, %lu elements", g_iterations, (unsigned long) num_elements);
	printf("\nCodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n\n",
		cuda_props.device_sm_version, cuda_props.kernel_ptx_version);

	printf(""
		"TuneID, "
		"SCHEDULING_GRANULARITY, "

		"UPSWEEP_LOG_THREADS, "
		"UPSWEEP_LOG_LOAD_VEC_SIZE, "
		"UPSWEEP_LOG_LOADS_PER_TILE, "

		"SPINE_LOG_THREADS, "
		"SPINE_LOG_LOAD_VEC_SIZE, "
		"SPINE_LOG_LOADS_PER_TILE, "

		"DOWNSWEEP_LOG_THREADS, "
		"DOWNSWEEP_LOG_LOAD_VEC_SIZE, "
		"DOWNSWEEP_LOG_LOADS_PER_TILE, "

		"elapsed time (ms), "
		"throughput (10^9 items/s), "
		"bandwidth (10^9 B/s)");
	if (g_verify) printf(", Correctness");
	printf("\n");


	// Execute test(s)
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 1)
	{
		typedef unsigned char T;
		Sum<T> reduction_op;
		Equality<T> equality_op;

		Test<T, T>(num_elements * 4, reduction_op, equality_op);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 2)
	{
		typedef unsigned short T;
		Sum<T> reduction_op;
		Equality<T> equality_op;
		Test<T, T>(num_elements * 2, reduction_op, equality_op);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 4)
	{
		typedef unsigned int T, T;
		Sum<T> reduction_op;
		Equality<T> equality_op;
		Test<T, T>(num_elements, reduction_op, equality_op);
	}
#endif
#if (TUNE_SIZE == 0) || (TUNE_SIZE == 8)
	{
		typedef unsigned long long T;
		Sum<T> reduction_op;
		Equality<T> equality_op;
		Test<T, T>(num_elements / 2, reduction_op, equality_op);
	}
#endif

	return 0;
}



