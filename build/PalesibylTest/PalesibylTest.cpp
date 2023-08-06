#include "pch.h"
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Palesibyl ;

namespace PalesibylTest
{
	TEST_CLASS(PalesibylTest)
	{
	public:
		
		// ピッタリサイズな行列×列ベクトル
		TEST_METHOD(TestMatrixVec16x8)
		{
			const size_t	dimSrc = 16 ;
			const size_t	dimDst = 8 ;

			NNMatrix	matrix ;
			matrix.Create( dimDst, dimSrc ) ;
			matrix.Randomize( -1.0f, 1.0f ) ;

			float	src[dimSrc] =
			{
				 1,  2,  3,  5,  7,  9, 11, 13,
				17, 19, 23, 29, 31, 37, 41, 43,
			} ;
			float	dst[dimDst] ;
			matrix.ProductVector( dst, src ) ;

			double	test = 0.0 ;
			for ( size_t i = 0; i < dimDst; i ++ )
			{
				float	d = 0.0f ;
				for ( size_t j = 0; j < dimSrc; j ++ )
				{
					d += matrix.At(i,j) * src[j] ;
				}
				test += fabs( dst[i] - d ) ;
			}
			Assert::IsTrue( test / (dimSrc * dimDst) < 1.0e-6 ) ;
		}

		// 半端サイズな行列×列ベクトル
		TEST_METHOD(TestMatrixVec15x7)
		{
			const size_t	dimSrc = 15 ;
			const size_t	dimDst = 7 ;

			NNMatrix	matrix ;
			matrix.Create( dimDst, dimSrc ) ;
			matrix.Randomize( -1.0f, 1.0f ) ;

			float	src[dimSrc] =
			{
				 1,  2,  3,  5,  7,  9, 11, 13,
				17, 19, 23, 29, 31, 37, 41,
			} ;
			float	dst[dimDst] ;
			matrix.ProductVector( dst, src ) ;

			double	test = 0.0 ;
			for ( size_t i = 0; i < dimDst; i ++ )
			{
				float	d = 0.0f ;
				for ( size_t j = 0; j < dimSrc; j ++ )
				{
					d += matrix.At(i,j) * src[j] ;
				}
				test += fabs( dst[i] - d ) ;
			}
			Assert::IsTrue( test / (dimSrc * dimDst) < 1.0e-6 ) ;
		}

		// NNBufDimCompareLess テスト
		TEST_METHOD(TestMapKeyAsBufDim)
		{
			std::map< NNBufDim,int,NNBufDimCompareLess >	map ;

			NNBufDim	dimTest1( 10, 20, 30 ) ;
			NNBufDim	dimTest2( 20, 30, 10 ) ;
			NNBufDim	dimTest3( 30, 10, 20 ) ;
			NNBufDim	dimTest4( 40, 30, 20 ) ;

			map.insert( std::make_pair( dimTest1, 1 ) ) ;
			map.insert( std::make_pair( dimTest2, 2 ) ) ;
			map.insert( std::make_pair( dimTest3, 3 ) ) ;
			map.insert( std::make_pair( dimTest4, 4 ) ) ;

			Assert::IsTrue( (map.find(dimTest1) != map.end())
							&& (map.find(dimTest1)->second == 1) ) ;
			Assert::IsTrue( (map.find(dimTest2) != map.end())
							&& (map.find(dimTest2)->second == 2) ) ;
			Assert::IsTrue( (map.find(dimTest3) != map.end())
							&& (map.find(dimTest3)->second == 3) ) ;
			Assert::IsTrue( (map.find(dimTest4) != map.end())
							&& (map.find(dimTest4)->second == 4) ) ;
		}

		// 線形回帰テスト（CPU・単一スレッド）
		TEST_METHOD(TestLinearRegressionWithCPU)
		{
			cudaDisable() ;

			const size_t	nCount = 1000 ;		// サンプル数
			const float		a = 0.78539816f ;	// 適当な f(x) = ax + b
			const float		b = -0.679570457f ;

			// 適当にばらついたサンプルを用意する
			std::random_device						random ;
			std::mt19937							engine( random() ) ;
			std::uniform_real_distribution<float>	distOut( -0.01f, 0.01f ) ;
			std::uniform_real_distribution<float>	distIn( -1.0f, 1.0f ) ;

			NNBuffer	bufTrainIn ;
			NNBuffer	bufTrainOut ;
			bufTrainIn.Create( nCount, 1, 1 ) ;
			bufTrainOut.Create( nCount, 1, 1 ) ;
			for ( size_t i = 0; i < nCount; i ++ )
			{
				float	x = distIn(engine) ;
				bufTrainIn.GetBuffer()[i] = x ;
				bufTrainOut.GetBuffer()[i] = (a * x + b) + distOut(engine) ;
			}

			// 単純パーセプトロン
			NNPerceptron	perceptron
				( 1, 1, 1, 1,
					std::make_shared<NNSamplerInjection>(),
					std::make_shared<NNActivationLinear>() ) ;

			// 処理用バッファ準備
			NNLoopStream				stream ;
			NNPerceptron::CPUWorkArray	works ;
			NNPerceptron::BufferArray	bufArray ;
			std::shared_ptr<NNPerceptron::Buffer>
										pBuf = std::make_shared<NNPerceptron::Buffer>() ;
			bufArray.push_back( pBuf ) ;
			perceptron.ResetBuffer( *pBuf, 0 ) ;
			perceptron.PrepareBuffer
				( *pBuf, bufTrainIn.GetSize(), bufArray, stream, 0,
					NNPerceptron::bufferForLearning ) ;
			perceptron.PrepareWorkArray
				( works, stream.m_ploop.GetThreadCount() ) ;

			// 回帰
			for ( int i = 0; i < 10000; i ++ )
			{
				perceptron.ResetWorkArrayInBatch( works ) ;
				NNPerceptron::InputBuffer
					inBuf = perceptron.PrepareInput( bufArray, 0, bufTrainIn, 0, stream ) ;
				perceptron.cpuPrediction( works, *pBuf, inBuf, stream ) ;
				perceptron.cpuLossDelta( works, *pBuf, bufTrainOut, stream ) ;
				perceptron.cpuCalcMatrixGradient( works, *pBuf, inBuf, stream ) ;
				perceptron.cpuIntegrateMatrixGradient( works, *pBuf ) ;
				perceptron.AddMatrixGradient( works, 0.01f ) ;
			}
			Assert::IsTrue( fabs(perceptron.GetMatrix().GetAt(0,0) - a) < 1.0e-2 ) ;
			Assert::IsTrue( fabs(perceptron.GetMatrix().GetAt(0,1) - b) < 1.0e-2 ) ;
		}

		// 線形回帰テスト（CUDA）（※CUDAが使えない場合CPU）
		TEST_METHOD(TestLinearRegressionWithCUDA)
		{
			cudaInit() ;

			const size_t	nCount = 1000 ;		// サンプル数
			const float		a = 0.78539816f ;	// 適当な f(x) = ax + b
			const float		b = -0.679570457f ;

			// 適当にばらついたサンプルを用意する
			std::random_device						random ;
			std::mt19937							engine( random() ) ;
			std::uniform_real_distribution<float>	distOut( -0.01f, 0.01f ) ;
			std::uniform_real_distribution<float>	distIn( -1.0f, 1.0f ) ;

			NNBuffer	bufTrainIn ;
			NNBuffer	bufTrainOut ;
			bufTrainIn.Create( nCount, 1, 1 ) ;
			bufTrainOut.Create( nCount, 1, 1 ) ;
			for ( size_t i = 0; i < nCount; i ++ )
			{
				float	x = distIn(engine) ;
				bufTrainIn.GetBuffer()[i] = x ;
				bufTrainOut.GetBuffer()[i] = (a * x + b) + distOut(engine) ;
			}

			// 単純パーセプトロン
			NNPerceptron	perceptron
				( 1, 1, 1, 1,
					std::make_shared<NNSamplerInjection>(),
					std::make_shared<NNActivationLinear>() ) ;

			// 処理用バッファ準備
			NNLoopStream				stream ;
			NNPerceptron::CPUWorkArray	works ;
			NNPerceptron::BufferArray	bufArray ;
			std::shared_ptr<NNPerceptron::Buffer>
										pBuf = std::make_shared<NNPerceptron::Buffer>() ;
			bufArray.push_back( pBuf ) ;
			perceptron.ResetBuffer( *pBuf, 0 ) ;
			perceptron.PrepareBuffer
				( *pBuf, bufTrainIn.GetSize(), bufArray, stream, 0,
					NNPerceptron::bufferForLearning ) ;
			perceptron.PrepareWorkArray
				( works, stream.m_ploop.GetThreadCount() ) ;

			// 回帰
			for ( int i = 0; i < 10000; i ++ )
			{
				perceptron.ResetWorkArrayInBatch( works ) ;
				NNPerceptron::InputBuffer
					inBuf = perceptron.PrepareInput( bufArray, 0, bufTrainIn, 0, stream ) ;
				perceptron.Prediction( works, *pBuf, inBuf, stream ) ;
				perceptron.LossDelta( works, *pBuf, bufTrainOut, stream ) ;
				perceptron.CalcMatrixGradient( works, *pBuf, inBuf, stream ) ;
				perceptron.IntegrateMatrixGradient( works, *pBuf, stream ) ;
				perceptron.AddMatrixGradient( works, 0.01f ) ;
			}
			Assert::IsTrue( fabs(perceptron.GetMatrix().GetAt(0,0) - a) < 1.0e-2 ) ;
			Assert::IsTrue( fabs(perceptron.GetMatrix().GetAt(0,1) - b) < 1.0e-2 ) ;
		}

	};
}
