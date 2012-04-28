

#include <stdio.h>

namespace cub {

#if defined(_MSC_VER)

	typedef long Spinlock;

	#include <intrin.h>

	/**
	 * Compiler read/write barrier
	 */
	#pragma intrinsic(_ReadWriteBarrier)

	/**
	 * Pause instruction to prevent excess processor bus usage
	 */
	__forceinline__ void CpuRelax()
	{
		__nop(); // YieldProcessor(); is reputed to generate pause instrs (instead of nops), but requires windows.h
	}

#else

	typedef int Spinlock;

	/**
	 * Compiler read/write barrier
	 */
	__forceinline__ void _ReadWriteBarrier()
	{
		__sync_synchronize();
	}

	/**
	 * Atomic exchange
	 */
	__forceinline__ long _InterlockedExchange(volatile int * const Target, const int Value)
	{
		// NOTE: __sync_lock_test_and_set would be an acquire barrier, so we force a full barrier
		_ReadWriteBarrier();
		return __sync_lock_test_and_set(Target, Value);
	}

	/**
	 * Pause instruction to prevent excess processor bus usage
	 */
	__forceinline__ void CpuRelax()
	{
		asm volatile("pause\n": : :"memory");
	}

#endif


/**
 * Return when the specified spinlock has been acquired
 */
__forceinline__ void Lock(volatile Spinlock *lock)
{
	while (1) {
		if (!_InterlockedExchange(lock, 1)) return;
		while(*lock) CpuRelax();
	}
}

/**
 * Release the specified spinlock
 */
__forceinline__ void Unlock(volatile Spinlock *lock)
{
	_ReadWriteBarrier();
	*lock = 0;
}

} // namespace cub


int main()
{
	using namespace cub;

	Spinlock lock(0);

	Lock(&lock);

	Unlock(&lock);

	printf("Done\n");
}
