

#if	defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
#if	defined(_DLL)
	#if defined(NDEBUG) || !defined(_DEBUG)
		#pragma	comment( lib, "palesibyl_md.lib" )
	#else
		#pragma	comment( lib, "palesibyl_mdb.lib" )
	#endif
#else
	#if defined(NDEBUG) || !defined(_DEBUG)
		#pragma	comment( lib, "palesibyl.lib" )
	#else
		#pragma	comment( lib, "palesibyl_db.lib" )
	#endif
#endif
#endif

