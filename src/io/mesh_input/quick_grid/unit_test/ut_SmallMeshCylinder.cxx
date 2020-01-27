//@HEADER
// ************************************************************************
//
//                             Peridigm
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?
// David J. Littlewood   djlittl@sandia.gov
// John A. Mitchell      jamitch@sandia.gov
// Michael L. Parks      mlparks@sandia.gov
// Stewart A. Silling    sasilli@sandia.gov
//
// ************************************************************************
//@HEADER

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include "Teuchos_UnitTestRepository.hpp"
#include "Vector3D.h"
#include "Array.h"
#include "../QuickGrid.h"
#include "PdutMpiFixture.h"
#include <iostream>


using std::tr1::shared_ptr;
using namespace Pdut;
using std::cout;

TEUCHOS_UNIT_TEST( SmallMeshCylinder, cylindericalCellPerProcIterator1ProcTest) {


	/*
	 * Construct ring spec
	 */
	// set ring at point = (0,0,0)
	// Note that ring is only translated to the location
	UTILITIES::Vector3D center;
	double ringThickness = 2.0;
	double innerRadius = ringThickness*(7/(2.0*M_PI)-1)*.5;
	double outerRadius = innerRadius+ringThickness;
	size_t numRings = 2;
	double cellSize=ringThickness/numRings;
	QUICKGRID::SpecRing2D ring2dSpec(center,innerRadius,outerRadius,numRings);
	QUICKGRID::Spec1D thetaSpec(ring2dSpec.getNumRays(), 0.0, 2.0*M_PI);
	/*
	 * Compute number of cells along length of cylinder so that aspect ratio
	 * is cells is approximately 1
	 * Use average radius to compute arc length of cell
	 *
	 */
	double cylinderLength = 2.0*cellSize;
	size_t numCellsAxis = 2;
	QUICKGRID::Spec1D axisSpec(numCellsAxis,0.0,cylinderLength);
	TEST_ASSERT(2==axisSpec.getNumCells());

	// This is a hack to get the correct number of cells in a ring
	double SCALE=1.51;
	double horizon = SCALE*cellSize;
	QUICKGRID::RingHorizon ringHorizon = thetaSpec.getRingCellHorizon(horizon,(innerRadius+outerRadius)/2.0);
	TEST_ASSERT(16 == ring2dSpec.getNumCells());
	/*
	 * Testing
	 */
	/*
	 * Check cells in ring horizon
	 */
	{
		int cell=0;
		QUICKGRID::RingHorizon::RingHorizonIterator hIter = ringHorizon.horizonIterator(cell);
		TEST_ASSERT(5 == hIter.numCells());
		TEST_ASSERT(hIter.hasNextCell());
		// loop over cells in horizon
		TEST_ASSERT(6 == hIter.nextCell());
		TEST_ASSERT(hIter.hasNextCell());
		TEST_ASSERT(7 == hIter.nextCell());
		TEST_ASSERT(hIter.hasNextCell());
		TEST_ASSERT(0 == hIter.nextCell());
		TEST_ASSERT(hIter.hasNextCell());
		TEST_ASSERT(1 == hIter.nextCell());
		TEST_ASSERT(hIter.hasNextCell());
		TEST_ASSERT(2 == hIter.nextCell());
		TEST_ASSERT(!hIter.hasNextCell());
	}

	UTILITIES::Array<double> rPtr = getDiscretization(ring2dSpec.getRaySpec());
	UTILITIES::Array<double> thetaPtr = getDiscretization(ring2dSpec.getRingSpec());
	UTILITIES::Array<double> zPtr = getDiscretization(axisSpec);

	double dr = ring2dSpec.getRaySpec().getCellSize();
	double dz = axisSpec.getCellSize();
	double cellRads = ring2dSpec.getRingSpec().getCellSize();

	// Create decomposition iterator
	size_t numProcs = 1;
	QUICKGRID::TensorProductCylinderMeshGenerator cellPerProcIter(numProcs, horizon,ring2dSpec, axisSpec);
	QUICKGRID::QuickGridData gridData;
	/*
	 * Testing
	 */
	{

		int i=0,j=0,k=0,proc=0;
		QUICKGRID::Cell3D cellLocator(i,j,k);
		QUICKGRID::RingHorizon::RingHorizonIterator hIter = ringHorizon.horizonIterator(j);

		QUICKGRID::QuickGridData pdGridDataProc0 = cellPerProcIter.allocatePdGridData();
		std::pair<QUICKGRID::Cell3D,QUICKGRID::QuickGridData> p0Data = cellPerProcIter.computePdGridData(proc,cellLocator,pdGridDataProc0);
		gridData = p0Data.second;
//		QUICKGRID::Cell3D nextCellLocator = p0Data.first;

		TEST_ASSERT(3==gridData.dimension);
		TEST_ASSERT(32==gridData.globalNumPoints);
		TEST_ASSERT(32==gridData.numPoints);
		TEST_ASSERT((19*32+32)==gridData.sizeNeighborhoodList);
		TEST_ASSERT(0==gridData.numExport);
		int *gIds = gridData.myGlobalIDs.get();
		for(size_t p=0;p<gridData.numPoints;p++,gIds++)
			TEST_ASSERT((int)p==*gIds);

		int neighborAnswers[] = {
				12,13,14,15,1,2,3,4,5,28,29,30,31,16,17,18,19,20,21,
				12,13,14,15,0,2,3,4,5,28,29,30,31,16,17,18,19,20,21,
				      14,15,0,1,3,4,5, 6, 7,30,31,16,17,18,19,20,21,22,23,
				      14,15,0,1,2,4,5, 6, 7,30,31,16,17,18,19,20,21,22,23,
				            0,1,2,3,5, 6, 7,8,9, 16,17,18,19,20,21,22,23,24,25,
				            0,1,2,3,4, 6, 7,8,9, 16,17,18,19,20,21,22,23,24,25,
				            2,3,4,5,7,8,9,10,11,18,19,20,21,22,23,24,25,26,27,
				            2,3,4,5,6,8,9,10,11,18,19,20,21,22,23,24,25,26,27,
				            4,5,6,7,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,
				            4,5,6,7,8,10,11,12,13,20,21,22,23,24,25,26,27,28,29,
				            6,7,8,9,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,
				            6,7,8,9,10,12,13,14,15,22,23,24,25,26,27,28,29,30,31,
				            8,9,10,11,13,14,15,0,1,24,25,26,27,28,29,30,31,16,17,
				            8,9,10,11,12,14,15,0,1,24,25,26,27,28,29,30,31,16,17,
				            10,11,12,13,15,0,1,2,3,26,27,28,29,30,31,16,17,18,19,
				            10,11,12,13,14,0,1,2,3,26,27,28,29,30,31,16,17,18,19,
				            // start new z-plane
							12,13,14,15,0,1,2,3,4,5,28,29,30,31,17,18,19,20,21,
							12,13,14,15,0,1,2,3,4,5,28,29,30,31,16,18,19,20,21,
							      14,15,0,1,2,3,4,5, 6, 7,30,31,16,17,19,20,21,22,23,
							      14,15,0,1,2,3,4,5, 6, 7,30,31,16,17,18,20,21,22,23,
							            0,1,2,3,4,5, 6, 7,8,9, 16,17,18,19,21,22,23,24,25,
							            0,1,2,3,4,5, 6, 7,8,9, 16,17,18,19,20,22,23,24,25,
							            2,3,4,5,6,7,8,9,10,11,18,19,20,21,23,24,25,26,27,
							            2,3,4,5,6,7,8,9,10,11,18,19,20,21,22,24,25,26,27,
							            4,5,6,7,8,9,10,11,12,13,20,21,22,23,25,26,27,28,29,
							            4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,26,27,28,29,
							            6,7,8,9,10,11,12,13,14,15,22,23,24,25,27,28,29,30,31,
							            6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,28,29,30,31,
							            8,9,10,11,12,13,14,15,0,1,24,25,26,27,29,30,31,16,17,
							            8,9,10,11,12,13,14,15,0,1,24,25,26,27,28,30,31,16,17,
							            10,11,12,13,14,15,0,1,2,3,26,27,28,29,31,16,17,18,19,
							            10,11,12,13,14,15,0,1,2,3,26,27,28,29,30,16,17,18,19

		};
		double *X = gridData.myX.get();
		double *r = rPtr.get();
		double *theta = thetaPtr.get();
		double *z = zPtr.get();
		int nx = ring2dSpec.getNumRings();
		int ny = thetaSpec.getNumCells();
		int *neighborhoodPtr = gridData.neighborhoodPtr.get();
		int *neighborhood = gridData.neighborhood.get();
		double *vol = gridData.cellVolume.get();

		for(size_t k=0;k<axisSpec.getNumCells();k++){
			double ranZ = z[k];
			for(size_t j=0;j<thetaSpec.getNumCells();j++){
				double ranTheta = theta[j];
				for(size_t i=0;i<ring2dSpec.getNumRings();i++){
					double ranR = r[i];
					double x = ranR*cos(ranTheta);
					double y = ranR*sin(ranTheta);
					double z = ranZ;
					int gId =  i + j * nx + k * nx * ny;
                    const double tolerance = 1.0e-13;
					TEST_FLOATING_EQUALITY(x, X[3*gId], tolerance);
					TEST_FLOATING_EQUALITY(y, X[3*gId+1], tolerance);
					TEST_FLOATING_EQUALITY(z, X[3*gId+2], tolerance);
					int ptr = neighborhoodPtr[gId];
					TEST_ASSERT(19==neighborhood[ptr]);
					for(int p=0;p<19;p++)
						TEST_ASSERT(neighborAnswers[p+gId*19]==neighborhood[ptr+1+p]);
					/*
					 * Volume
					 */
					double v =  ranR*dr*cellRads*dz;
					TEST_FLOATING_EQUALITY(v, vol[gId], tolerance);
				}
			}
		}
	}

}


TEUCHOS_UNIT_TEST( SmallMeshCylinder, cylindericalCellPerProcIterator2ProcTest) {


	/*
	 * Construct ring spec
	 */
	// set ring at point = (0,0,0)
	// Note that ring is only translated to the location
	UTILITIES::Vector3D center;
	double ringThickness = 2.0;
	double innerRadius = ringThickness*(7/(2.0*M_PI)-1)*.5;
	double outerRadius = innerRadius+ringThickness;
	size_t numRings = 2;
	double cellSize=ringThickness/numRings;
	QUICKGRID::SpecRing2D ring2dSpec(center,innerRadius,outerRadius,numRings);
	QUICKGRID::Spec1D thetaSpec(ring2dSpec.getNumRays(), 0.0, 2.0*M_PI);
	/*
	 * Compute number of cells along length of cylinder so that aspect ratio
	 * is cells is approximately 1
	 * Use average radius to compute arc length of cell
	 *
	 */
	double cylinderLength = 2.0*cellSize;
	size_t numCellsAxis = 2;
	QUICKGRID::Spec1D axisSpec(numCellsAxis,0.0,cylinderLength);
	TEST_ASSERT(2==axisSpec.getNumCells());

	// This is a hack to get the correct number of cells in a ring
	double SCALE=1.51;
	double horizon = SCALE*cellSize;

	UTILITIES::Array<double> rPtr = getDiscretization(ring2dSpec.getRaySpec());
	UTILITIES::Array<double> thetaPtr = getDiscretization(ring2dSpec.getRingSpec());
	UTILITIES::Array<double> zPtr = getDiscretization(axisSpec);

	double dr = ring2dSpec.getRaySpec().getCellSize();
	double dz = axisSpec.getCellSize();
	double cellRads = ring2dSpec.getRingSpec().getCellSize();

	int neighborAnswers[] = {
			12,13,14,15,1,2,3,4,5,28,29,30,31,16,17,18,19,20,21,
			12,13,14,15,0,2,3,4,5,28,29,30,31,16,17,18,19,20,21,
			      14,15,0,1,3,4,5, 6, 7,30,31,16,17,18,19,20,21,22,23,
			      14,15,0,1,2,4,5, 6, 7,30,31,16,17,18,19,20,21,22,23,
			            0,1,2,3,5, 6, 7,8,9, 16,17,18,19,20,21,22,23,24,25,
			            0,1,2,3,4, 6, 7,8,9, 16,17,18,19,20,21,22,23,24,25,
			            2,3,4,5,7,8,9,10,11,18,19,20,21,22,23,24,25,26,27,
			            2,3,4,5,6,8,9,10,11,18,19,20,21,22,23,24,25,26,27,
			            4,5,6,7,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,
			            4,5,6,7,8,10,11,12,13,20,21,22,23,24,25,26,27,28,29,
			            6,7,8,9,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,
			            6,7,8,9,10,12,13,14,15,22,23,24,25,26,27,28,29,30,31,
			            8,9,10,11,13,14,15,0,1,24,25,26,27,28,29,30,31,16,17,
			            8,9,10,11,12,14,15,0,1,24,25,26,27,28,29,30,31,16,17,
			            10,11,12,13,15,0,1,2,3,26,27,28,29,30,31,16,17,18,19,
			            10,11,12,13,14,0,1,2,3,26,27,28,29,30,31,16,17,18,19,
			            // start new z-plane
						12,13,14,15,0,1,2,3,4,5,28,29,30,31,17,18,19,20,21,
						12,13,14,15,0,1,2,3,4,5,28,29,30,31,16,18,19,20,21,
						      14,15,0,1,2,3,4,5, 6, 7,30,31,16,17,19,20,21,22,23,
						      14,15,0,1,2,3,4,5, 6, 7,30,31,16,17,18,20,21,22,23,
						            0,1,2,3,4,5, 6, 7,8,9, 16,17,18,19,21,22,23,24,25,
						            0,1,2,3,4,5, 6, 7,8,9, 16,17,18,19,20,22,23,24,25,
						            2,3,4,5,6,7,8,9,10,11,18,19,20,21,23,24,25,26,27,
						            2,3,4,5,6,7,8,9,10,11,18,19,20,21,22,24,25,26,27,
						            4,5,6,7,8,9,10,11,12,13,20,21,22,23,25,26,27,28,29,
						            4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,26,27,28,29,
						            6,7,8,9,10,11,12,13,14,15,22,23,24,25,27,28,29,30,31,
						            6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,28,29,30,31,
						            8,9,10,11,12,13,14,15,0,1,24,25,26,27,29,30,31,16,17,
						            8,9,10,11,12,13,14,15,0,1,24,25,26,27,28,30,31,16,17,
						            10,11,12,13,14,15,0,1,2,3,26,27,28,29,31,16,17,18,19,
						            10,11,12,13,14,15,0,1,2,3,26,27,28,29,30,16,17,18,19

	};

	// Create decomposition iterator
	size_t numProcs = 2;
	QUICKGRID::TensorProductCylinderMeshGenerator cellIter(numProcs, horizon,ring2dSpec, axisSpec);
	QUICKGRID::QuickGridData gridData;
	/*
	 * Testing
	 */
	{
		QUICKGRID::QuickGridData pdGridDataProc0 = cellIter.allocatePdGridData();
		QUICKGRID::QuickGridData  pdGridDataProcN = cellIter.allocatePdGridData();
		std::pair<QUICKGRID::Cell3D,QUICKGRID::QuickGridData> p0Data = cellIter.beginIterateProcs(pdGridDataProc0);

		QUICKGRID::QuickGridData gridData = p0Data.second;
		QUICKGRID::Cell3D nextCellLocator = p0Data.first;
		// proc 0
		TEST_ASSERT(3 == gridData.dimension);
		TEST_ASSERT(32 == gridData.globalNumPoints);
		int myNumPoints = gridData.numPoints;
		TEST_ASSERT(16 == myNumPoints);

		// assert length of neighborlist
		// sizeNeighborList = myNumCells + myNumCells*numNeighbors
		int sizeNeighborList = myNumPoints + myNumPoints*19;
		TEST_ASSERT( sizeNeighborList == gridData.sizeNeighborhoodList );
		TEST_ASSERT(0 == gridData.numExport);

		// Assert global ids for this processor
		shared_ptr<int> gIds = gridData.myGlobalIDs;
		int *gIdsPtr = gIds.get();
		int start = 0;
		for(size_t id=start;id<gridData.numPoints+start;id++,gIdsPtr++)
			TEST_ASSERT( *gIdsPtr == (int)id );

		{
			// Assert coordinates and volume
			double *X = gridData.myX.get();
			double *r = rPtr.get();
			double *theta = thetaPtr.get();
			double *z = zPtr.get();
			int *neighborhoodPtr = gridData.neighborhoodPtr.get();
			int *neighborhood = gridData.neighborhood.get();
			double *vol = gridData.cellVolume.get();

			int k=0;
			int cell=0;
			double ranZ = z[k];
			for(size_t j=0;j<thetaSpec.getNumCells();j++){
				double ranTheta = theta[j];
				for(size_t i=0;i<ring2dSpec.getNumRings();i++){
					double ranR = r[i];
					double x = ranR*cos(ranTheta);
					double y = ranR*sin(ranTheta);
					double z = ranZ;
                    const double tolerance = 1.0e-13;
					TEST_FLOATING_EQUALITY(x, X[3*cell], tolerance);
					TEST_FLOATING_EQUALITY(y, X[3*cell+1], tolerance);
					TEST_FLOATING_EQUALITY(z, X[3*cell+2], tolerance);
					int ptr = neighborhoodPtr[cell];
					TEST_ASSERT(19==neighborhood[ptr]);
					for(int p=0;p<19;p++){
						TEST_ASSERT(neighborAnswers[p+(cell)*19]==neighborhood[ptr+1+p]);
					}
					/*
					 * Volume
					 */
					double v =  ranR*dr*cellRads*dz;
					TEST_FLOATING_EQUALITY(v, vol[cell], tolerance);
					cell++;
				}
			}

		}
		// already moved to next proc
		size_t proc = 1;
		start=16;
		while(cellIter.hasNextProc()){

			TEST_ASSERT(proc == cellIter.proc());
			std::pair<QUICKGRID::Cell3D,QUICKGRID::QuickGridData> data = cellIter.nextProc(nextCellLocator,pdGridDataProcN);

			QUICKGRID::QuickGridData gridData = data.second;
			nextCellLocator = data.first;

			TEST_ASSERT(3 == gridData.dimension);
			TEST_ASSERT(32 == gridData.globalNumPoints);
			int myNumPoints = gridData.numPoints;
			TEST_ASSERT(16 == myNumPoints);

			// assert length of neighborlist
			// sizeNeighborList = myNumCells + myNumCells*numNeighbors
			int sizeNeighborList = myNumPoints + myNumPoints*19;
			TEST_ASSERT( sizeNeighborList == gridData.sizeNeighborhoodList );
			TEST_ASSERT(0 == gridData.numExport);

			// assert global ids for this processor
			shared_ptr<int> gIds = gridData.myGlobalIDs;
			int *gIdsPtr = gIds.get();
			for(size_t id=start;id<gridData.numPoints+start;id++,gIdsPtr++){
				TEST_ASSERT( *gIdsPtr == (int)id );
			}

			// Assert coordinates
			double *X = gridData.myX.get();
			double *r = rPtr.get();
			double *theta = thetaPtr.get();
			double *z = zPtr.get();
			int *neighborhoodPtr = gridData.neighborhoodPtr.get();
			int *neighborhood = gridData.neighborhood.get();
			double *vol = gridData.cellVolume.get();

			int k=1;
			int cell=0;
			double ranZ = z[k];
			for(size_t j=0;j<thetaSpec.getNumCells();j++){
				double ranTheta = theta[j];
				for(size_t i=0;i<ring2dSpec.getNumRings();i++){
					double ranR = r[i];
					double x = ranR*cos(ranTheta);
					double y = ranR*sin(ranTheta);
					double z = ranZ;
                    const double tolerance = 1.0e-13;
                    TEST_FLOATING_EQUALITY(x, X[3*cell], tolerance);
					TEST_FLOATING_EQUALITY(y, X[3*cell+1], tolerance);
					TEST_FLOATING_EQUALITY(z, X[3*cell+2], tolerance);
					int ptr = neighborhoodPtr[cell];
					TEST_ASSERT(19==neighborhood[ptr]);
					for(int p=0;p<19;p++){
						TEST_ASSERT(neighborAnswers[p+(cell+16)*19]==neighborhood[ptr+1+p]);
					}
					/*
					 * Volume
					 */
					double v =  ranR*dr*cellRads*dz;
					TEST_FLOATING_EQUALITY(v, vol[cell], tolerance);
					cell++;
				}
			}

			// there are 16 nodes per processor
			start += 16;

			proc++;

		}

	}

}



TEUCHOS_UNIT_TEST( SmallMeshCylinder, cylindericalCellPerProcIterator4ProcTest) 
{

	/*
	 * Construct ring spec
	 */
	// set ring at point = (0,0,0)
	// Note that ring is only translated to the location
	UTILITIES::Vector3D center;
	double ringThickness = 2.0;
	double innerRadius = ringThickness*(7/(2.0*M_PI)-1)*.5;
	double outerRadius = innerRadius+ringThickness;
	int numRings = 2;
	double cellSize=ringThickness/numRings;
	QUICKGRID::SpecRing2D ring2dSpec(center,innerRadius,outerRadius,numRings);
	QUICKGRID::Spec1D thetaSpec(ring2dSpec.getNumRays(), 0.0, 2.0*M_PI);
	/*
	 * Compute number of cells along length of cylinder so that aspect ratio
	 * is cells is approximately 1
	 * Use average radius to compute arc length of cell
	 *
	 */
	double cylinderLength = 2.0*cellSize;
	int numCellsAxis = 2;
	QUICKGRID::Spec1D axisSpec(numCellsAxis,0.0,cylinderLength);
	TEST_ASSERT(2==axisSpec.getNumCells());

	// This is a hack to get the correct number of cells in a ring
	double SCALE=1.51;
	double horizon = SCALE*cellSize;

	UTILITIES::Array<double> rPtr = getDiscretization(ring2dSpec.getRaySpec());
	UTILITIES::Array<double> thetaPtr = getDiscretization(ring2dSpec.getRingSpec());
	UTILITIES::Array<double> zPtr = getDiscretization(axisSpec);

	double dr = ring2dSpec.getRaySpec().getCellSize();
	double dz = axisSpec.getCellSize();
	double cellRads = ring2dSpec.getRingSpec().getCellSize();

	int neighborAnswers[] = {
			12,13,14,15,1,2,3,4,5,28,29,30,31,16,17,18,19,20,21,
			12,13,14,15,0,2,3,4,5,28,29,30,31,16,17,18,19,20,21,
			      14,15,0,1,3,4,5, 6, 7,30,31,16,17,18,19,20,21,22,23,
			      14,15,0,1,2,4,5, 6, 7,30,31,16,17,18,19,20,21,22,23,
			            0,1,2,3,5, 6, 7,8,9, 16,17,18,19,20,21,22,23,24,25,
			            0,1,2,3,4, 6, 7,8,9, 16,17,18,19,20,21,22,23,24,25,
			            2,3,4,5,7,8,9,10,11,18,19,20,21,22,23,24,25,26,27,
			            2,3,4,5,6,8,9,10,11,18,19,20,21,22,23,24,25,26,27,
			            4,5,6,7,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29,
			            4,5,6,7,8,10,11,12,13,20,21,22,23,24,25,26,27,28,29,
			            6,7,8,9,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,
			            6,7,8,9,10,12,13,14,15,22,23,24,25,26,27,28,29,30,31,
			            8,9,10,11,13,14,15,0,1,24,25,26,27,28,29,30,31,16,17,
			            8,9,10,11,12,14,15,0,1,24,25,26,27,28,29,30,31,16,17,
			            10,11,12,13,15,0,1,2,3,26,27,28,29,30,31,16,17,18,19,
			            10,11,12,13,14,0,1,2,3,26,27,28,29,30,31,16,17,18,19,
			            // start new z-plane
						12,13,14,15,0,1,2,3,4,5,28,29,30,31,17,18,19,20,21,
						12,13,14,15,0,1,2,3,4,5,28,29,30,31,16,18,19,20,21,
						      14,15,0,1,2,3,4,5, 6, 7,30,31,16,17,19,20,21,22,23,
						      14,15,0,1,2,3,4,5, 6, 7,30,31,16,17,18,20,21,22,23,
						            0,1,2,3,4,5, 6, 7,8,9, 16,17,18,19,21,22,23,24,25,
						            0,1,2,3,4,5, 6, 7,8,9, 16,17,18,19,20,22,23,24,25,
						            2,3,4,5,6,7,8,9,10,11,18,19,20,21,23,24,25,26,27,
						            2,3,4,5,6,7,8,9,10,11,18,19,20,21,22,24,25,26,27,
						            4,5,6,7,8,9,10,11,12,13,20,21,22,23,25,26,27,28,29,
						            4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,26,27,28,29,
						            6,7,8,9,10,11,12,13,14,15,22,23,24,25,27,28,29,30,31,
						            6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,28,29,30,31,
						            8,9,10,11,12,13,14,15,0,1,24,25,26,27,29,30,31,16,17,
						            8,9,10,11,12,13,14,15,0,1,24,25,26,27,28,30,31,16,17,
						            10,11,12,13,14,15,0,1,2,3,26,27,28,29,31,16,17,18,19,
						            10,11,12,13,14,15,0,1,2,3,26,27,28,29,30,16,17,18,19

	};

	// Create decomposition iterator
	size_t numProcs = 4;
	QUICKGRID::TensorProductCylinderMeshGenerator cellIter(numProcs, horizon,ring2dSpec, axisSpec);
	QUICKGRID::QuickGridData gridData;
	UTILITIES::Array<double> meshPtr = QUICKGRID::getDiscretization(ring2dSpec,axisSpec);
	QUICKGRID::QuickGridData pdGridDataProc0 = cellIter.allocatePdGridData();
	QUICKGRID::QuickGridData  pdGridDataProcN = cellIter.allocatePdGridData();
	QUICKGRID::Cell3D nextCellLocator(0,0,0);
	/*
	 * Testing
	 */
	{
		std::pair<QUICKGRID::Cell3D,QUICKGRID::QuickGridData> p0Data = cellIter.beginIterateProcs(pdGridDataProc0);

		QUICKGRID::QuickGridData gridData = p0Data.second;
		nextCellLocator = p0Data.first;

		// proc 0
		TEST_ASSERT(3 == gridData.dimension);
		TEST_ASSERT(32 == gridData.globalNumPoints);
		int myNumPoints = gridData.numPoints;
		TEST_ASSERT(8 == myNumPoints);

		// assert length of neighborlist
		// sizeNeighborList = myNumCells + myNumCells*numNeighbors
		int sizeNeighborList = myNumPoints + myNumPoints*19;
		TEST_ASSERT( sizeNeighborList == gridData.sizeNeighborhoodList );
		TEST_ASSERT(0 == gridData.numExport);


		double *X = gridData.myX.get();
		int *neighborhoodPtr = gridData.neighborhoodPtr.get();
		int *neighborhood = gridData.neighborhood.get();
		double *vol = gridData.cellVolume.get();
		double *xx = meshPtr.get();
		// Assert global ids for this processor
		shared_ptr<int> gIds = gridData.myGlobalIDs;
		int *gIdsPtr = gIds.get();
		int cell = 0;
		int start=0;
        const double tolerance = 1.0e-13;
		for(size_t id=start;id<gridData.numPoints+start;id++,gIdsPtr++,cell++){
			TEST_ASSERT( *gIdsPtr == (int)id );

			TEST_FLOATING_EQUALITY(xx[id*3], X[3*cell], tolerance);
			TEST_FLOATING_EQUALITY(xx[id*3+1], X[3*cell+1], tolerance);
			TEST_FLOATING_EQUALITY(xx[id*3+2], X[3*cell+2], tolerance);
			int ptr = neighborhoodPtr[cell];
			TEST_ASSERT(19==neighborhood[ptr]);
			for(int p=0;p<19;p++){
				TEST_ASSERT(neighborAnswers[p+id*19]==neighborhood[ptr+1+p]);
			}
			double r = sqrt(xx[id*3]*xx[id*3]+xx[id*3+1]*xx[id*3+1]);
			/*
			 * Volume
			 */
			double v = r*dr*cellRads*dz;
			TEST_FLOATING_EQUALITY(v,vol[cell],tolerance);
		}
	}

	{
		// already moved to next proc
		size_t proc = 1;
		int start=8;
		while(cellIter.hasNextProc()){
			TEST_ASSERT(proc == cellIter.proc());
			std::pair<QUICKGRID::Cell3D,QUICKGRID::QuickGridData> data = cellIter.nextProc(nextCellLocator,pdGridDataProcN);

			QUICKGRID::QuickGridData gridData = data.second;
			nextCellLocator = data.first;

			TEST_ASSERT(3 == gridData.dimension);
			TEST_ASSERT(32 == gridData.globalNumPoints);
			int myNumPoints = gridData.numPoints;
			TEST_ASSERT(8 == myNumPoints);

			// assert length of neighborlist
			// sizeNeighborList = myNumCells + myNumCells*numNeighbors
			int sizeNeighborList = myNumPoints + myNumPoints*19;
			TEST_ASSERT( sizeNeighborList == gridData.sizeNeighborhoodList );
			TEST_ASSERT(0 == gridData.numExport);


			double *X = gridData.myX.get();
			int *neighborhoodPtr = gridData.neighborhoodPtr.get();
			int *neighborhood = gridData.neighborhood.get();
			double *vol = gridData.cellVolume.get();
			double *xx = meshPtr.get();
			// Assert global ids for this processor
			shared_ptr<int> gIds = gridData.myGlobalIDs;
			int *gIdsPtr = gIds.get();
			int cell = 0;
            const double tolerance = 1.0e-13;
			for(size_t id=start;id<gridData.numPoints+start;id++,gIdsPtr++,cell++){
				TEST_ASSERT( *gIdsPtr == (int)id );

                TEST_FLOATING_EQUALITY(xx[id*3], X[3*cell], tolerance);
                TEST_FLOATING_EQUALITY(xx[id*3+1], X[3*cell+1], tolerance);
                TEST_FLOATING_EQUALITY(xx[id*3+2], X[3*cell+2], tolerance);
				int ptr = neighborhoodPtr[cell];
				TEST_ASSERT(19==neighborhood[ptr]);
				for(int p=0;p<19;p++){
					TEST_ASSERT(neighborAnswers[p+id*19]==neighborhood[ptr+1+p]);
				}
				double r = sqrt(xx[id*3]*xx[id*3]+xx[id*3+1]*xx[id*3+1]);
				/*
				 * Volume
				 */
				double v = r*dr*cellRads*dz;
				TEST_FLOATING_EQUALITY(v,vol[cell],tolerance);
			}

			// there are 16 nodes per processor
			start += 8;

			proc++;

		}

	}

}



int main
(
		int argc,
		char* argv[]
)
{

	// Initialize UTF
	return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
}

