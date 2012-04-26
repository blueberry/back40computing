

#include <stdio.h>

#if defined(_MSC_VER)

	typedef long Spinlock;

	#include <intrin.h>
	#pragma intrinsic(_ReadWriteBarrier)

#else

	typedef int Spinlock;

	__forceinline__ long _InterlockedExchange(volatile int * const Target, const int Value)
	{
		// NOTE: __sync_lock_test_and_set would be an acquire barrier, so we force a full barrier
		__sync_synchronize();
		return __sync_lock_test_and_set(Target, Value);
	}

	__forceinline__ void _ReadWriteBarrier() {
		__sync_synchronize();
	}

#endif



__forceinline__ void Lock(Spinlock *lock)
{
	while (1)
	{
		if (!_InterlockedExchange(lock, 1)) return;
	}
}

__forceinline__ void Unlock(Spinlock *lock)
{
	_ReadWriteBarrier();
	*lock = 0;
}


int main()
{
	Spinlock lock(0);

	Lock(&lock);

	Unlock(&lock);

	printf("Done\n");
}
