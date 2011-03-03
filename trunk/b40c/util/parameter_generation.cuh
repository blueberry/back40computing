/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Functionality for generating parameter lists based upon ranges, suitable
 * for auto-tuning
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {


/**
 * A recursive tuple type that wraps a constant integer and another tuple type.
 *
 * Can be used to construct types that describe static lists of integer constants.
 */
template <typename NextTuple, int N, int PARAM>
struct ParamTuple
{
	typedef NextTuple next;
	enum {
		P = PARAM,
		V = N
	};
};


template <typename ParamTuple, int SEARCH_PARAM, int PARAM = ParamTuple::P>
struct Access
{
	enum { VALUE = Access<typename ParamTuple::next, SEARCH_PARAM>::VALUE };
};

template <typename ParamTuple, int PARAM>
struct Access<ParamTuple, PARAM, PARAM>
{
	enum { VALUE = ParamTuple::V };
};


/**
 * A type generator that sweeps an enumerated sequence of tuning parameters,
 * each of which has an associated (integer) range.  A static list of integer
 * constants is generated for every possible permutation of parameter values.
 * A static callback function on the problem-description type TuneProblemDetail
 * is invoked for each permutation.
 *
 * The range structure for a given parameter may be dependent upon the
 * values selected for tuning parameters occurring prior in the
 * enumeration. (E.g., the range structure for a "raking threads" parameter
 * may incorporate a "cta threads" parameter that is swept earlier in
 * the enumeration to establish an upper bound on raking threads.)
 */
template <
	typename TuneProblemDetail,
	int PARAM,
	int MAX_PARAM,
	template <typename, int> class Ranges>
struct ParamListSweep
{
	// Next parameter increment
	template <int COUNT, int MAX>
	struct Sweep
	{
		template <typename ParamList>
		static void Invoke(TuneProblemDetail &detail)
		{
			// Sweep subsequent parameter
			ParamListSweep<
				TuneProblemDetail,
				PARAM + 1,
				MAX_PARAM,
				Ranges>::template Invoke<ParamTuple<ParamList, COUNT, PARAM> >(detail);

			// Continue sweep with increment of this parameter
			Sweep<COUNT + 1, MAX>::template Invoke<ParamList>(detail);
		}
	};

	// Terminate
	template <int MAX>
	struct Sweep<MAX, MAX>
	{
		template <typename ParamList>
		static void Invoke(TuneProblemDetail &detail) {}
	};

	// Interface
	template <typename ParamList>
	static void Invoke(TuneProblemDetail &detail)
	{
		// Sweep current parameter
		Sweep<
			Ranges<ParamList, PARAM>::MIN,
			Ranges<ParamList, PARAM>::MAX + 1>::template Invoke<ParamList>(detail);

	}
};

// End of currently-generated list
template <
	typename TuneProblemDetail,
	int MAX_PARAM,
	template <typename, int> class Ranges>
struct ParamListSweep <TuneProblemDetail, MAX_PARAM, MAX_PARAM, Ranges>
{
	template <typename ParamList>
	static void Invoke(TuneProblemDetail &detail)
	{
		// Invoke callback
		detail.template Invoke<ParamList>();
	}

};


} // namespace util
} // namespace b40c

