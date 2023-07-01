
#ifndef	__NN_SIMD_UTIL_H__
#define	__NN_SIMD_UTIL_H__

// ※ 無効化する場合には __NN_INHIBIT_SIMD__ を定義しておく

#ifndef	__NN_INHIBIT_SIMD__

#if	defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64) || (_M_IX86_FP >= 2)

	#define	__NN_USE_SIMD__
	#include <xmmintrin.h>
	#include <emmintrin.h>
	#include <intrin.h>

#endif

#endif


#endif

