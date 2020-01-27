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
#include "Teuchos_GlobalMPISession.hpp"
#include <iostream>
#include <cmath>
#include <map>
#include <set>

#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#include "mpi.h"
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Array.h"
#include "../PdZoltan.h"
#include "../BondFilter.h"
#include "../NeighborhoodList.h"
#include "quick_grid/QuickGrid.h"

#include "PdutMpiFixture.h"


using std::tr1::shared_ptr;

using namespace Pdut;
using std::cout;
using std::set;
using std::map;


const size_t nx = 4;
const size_t ny = 4;
const size_t nz = 1;
const double xStart = -.125;
const double xLength = 1.0;
const double yStart = -.125;
const double yLength = 1.0;
const double zStart = -0.5;
const double zLength = 1.0;
const QUICKGRID::Spec1D xSpec(nx,xStart,xLength);
const QUICKGRID::Spec1D ySpec(ny,yStart,yLength);
const QUICKGRID::Spec1D zSpec(nz,zStart,zLength);
const size_t numCells = nx*ny*nz;
const double horizon = sqrt(2)*xLength/nx;

bool init = false;
Teuchos::RCP<Epetra_Comm> comm;   


void initialize(){
         
       #ifdef HAVE_MPI
              
              comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
       #else
              comm = rcp(new Epetra_SerialComm);
       #endif
     
       
       init = true;
}


QUICKGRID::QuickGridData getGrid( int numProcs, int myRank) {
	QUICKGRID::TensorProduct3DMeshGenerator cellPerProcIter(numProcs,horizon,xSpec,ySpec,zSpec,QUICKGRID::SphericalNorm);
	QUICKGRID::QuickGridData decomp =  QUICKGRID::getDiscretization(myRank, cellPerProcIter);

	// This load-balances
	decomp = PDNEIGH::getLoadBalancedDiscretization(decomp);
	return decomp;
}

TEUCHOS_UNIT_TEST(ReloadBalance_np4, AssertOriginalMesh) {

       if (!init) initialize();

       int numProcs = comm->NumProc();
       int myRank   = comm->MyPID();


       TEST_COMPARE(numProcs, ==, 4);

       if(numProcs != 4){
          std::cerr << "Unit test runtime ERROR: utReloadBalance_np4 only makes sense on 4 processors." << std::endl;
          return;
      }

      

	QUICKGRID::QuickGridData decomp = getGrid( numProcs, myRank);
	TEST_ASSERT(16==decomp.globalNumPoints);

	if(1==myRank){
		size_t numPoints = decomp.numPoints;
		TEST_ASSERT(numPoints==decomp.numPoints);
		TEST_ASSERT(25==decomp.sizeNeighborhoodList);
		int myIds[] = {0,1,4,5};
		set<int> ids(myIds,myIds+numPoints);
		set<int>::iterator i=ids.begin();
		set<int>::iterator iEnd=ids.end();
		int *gIds = decomp.myGlobalIDs.get();
		for(size_t p=0;p<numPoints;p++,gIds++){
//			cout << "myRank, gIds, localId = " << myRank << ", " << *gIds << ", " << p << endl;
			TEST_ASSERT(iEnd!=ids.find(*gIds));
		}
	}
	if(0==myRank){
		size_t numPoints = decomp.numPoints;
		TEST_ASSERT(numPoints==decomp.numPoints);
		TEST_ASSERT(25==decomp.sizeNeighborhoodList);
		int myIds[] = {2,3,6,7};
		set<int> ids(myIds,myIds+numPoints);
		set<int>::iterator i=ids.begin();
		set<int>::iterator iEnd=ids.end();
		int *gIds = decomp.myGlobalIDs.get();
		for(size_t p=0;p<numPoints;p++,gIds++){
			TEST_ASSERT(iEnd!=ids.find(*gIds));
		}

	}
	if(2==myRank){
		size_t numPoints = decomp.numPoints;
		TEST_ASSERT(numPoints==decomp.numPoints);
		TEST_ASSERT(25==decomp.sizeNeighborhoodList);
		int myIds[] = {8,9,12,13};
		set<int> ids(myIds,myIds+numPoints);
		set<int>::iterator i=ids.begin();
		set<int>::iterator iEnd=ids.end();
		int *gIds = decomp.myGlobalIDs.get();
		for(size_t p=0;p<numPoints;p++,gIds++){
			TEST_ASSERT(iEnd!=ids.find(*gIds));
		}

	}
	if(3==myRank){
		size_t numPoints = decomp.numPoints;
		TEST_ASSERT(numPoints==decomp.numPoints);
		TEST_ASSERT(25==decomp.sizeNeighborhoodList);
		int myIds[] = {10,11,14,15};
		set<int> ids(myIds,myIds+numPoints);
		set<int>::iterator i=ids.begin();
		set<int>::iterator iEnd=ids.end();
		int *gIds = decomp.myGlobalIDs.get();
		for(size_t p=0;p<numPoints;p++,gIds++){
//			cout << "myRank, gIds, localId = " << myRank << ", " << *gIds << ", " << p << endl;
			TEST_ASSERT(iEnd!=ids.find(*gIds));
		}
	}

}

TEUCHOS_UNIT_TEST(ReloadBalance_np4, MoveCoordinatesAndReLoadBalance) {

       if (!init) initialize();

       int numProcs = comm->NumProc();
       int myRank   = comm->MyPID();


       TEST_COMPARE(numProcs, ==, 4);

       if(numProcs != 4){
          std::cerr << "Unit test runtime ERROR: utReloadBalance_np4 only makes sense on 4 processors." << std::endl;
          return;
      }

	QUICKGRID::QuickGridData decomp = getGrid( numProcs, myRank);
	/*
	 * Swap some coordinates to that new partition is required
	 * Swap 0 with 15, 3 with 12
	 */
	double *x = decomp.myX.get();
	if(1==myRank){
		/*
		 * point 0 has localId=2
		 */
		int localId0 = 2;
		double *x0 = x+3*localId0;
		double x15 = xStart + xLength - xLength/nx/2;
		double y15 = yStart + yLength - yLength/ny/2;
		double z15 = zStart + zLength - zLength/nz/2;
		*(x0+0)=x15;
		*(x0+1)=y15;
		*(x0+2)=z15;
	}
	if(3==myRank){
		/*
		 * point 15 has localId=1
		 */
		int localId0 = 1;
		double *x15 = x+3*localId0;
		double x0 = xStart + xLength/nx/2;
		double y0 = yStart + yLength/ny/2;
		double z0 = zStart + zLength/nz/2;
		*(x15+0)=x0;
		*(x15+1)=y0;
		*(x15+2)=z0;
	}

	/*
	 * Re-set neighborhood list
	 * 1) Set sizeNeighborhoodList=1
	 * 2) Create a new and empty neighborhood
	 * 3) Set neighborhood pointer for each point to 0
	 */
	decomp.sizeNeighborhoodList=1;
	UTILITIES::Array<int> neighborhoodList(decomp.sizeNeighborhoodList);
	decomp.neighborhood = neighborhoodList.get_shared_ptr();
	int *neighborhood = neighborhoodList.get();
	/*
	 * number of neighbors for every point is zero
	 */
	*neighborhood = 0;

	/*
	 * Re-set neighborhood pointer to point to first entry in list above
	 */
	int *neighPtr = decomp.neighborhoodPtr.get();
	for(size_t p=0;p<decomp.numPoints;p++,neighPtr++)
		*neighPtr=0;

	/*
	 * 1) Re-load balance
	 * 2) Re-compute neighborhood list
	 */
	decomp = PDNEIGH::getLoadBalancedDiscretization(decomp);
	shared_ptr<PdBondFilter::BondFilter> bondFilterPtr(new PdBondFilter::BondFilterDefault());
	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	std::vector< shared_ptr<PdBondFilter::BondFilter> > bondFilterPtrs;
	bondFilterPtrs.push_back(bondFilterPtr);
	PDNEIGH::NeighborhoodList list(comm,decomp.zoltanPtr.get(),decomp.numPoints,decomp.myGlobalIDs,decomp.myX,horizon,bondFilterPtrs);

	/*
	 * Now check that new partition has swapped points and
	 * that neighborhood lists are of the proper dimension
	 */
	if(1==myRank){
		int numPoints = decomp.numPoints;
		TEST_ASSERT(numPoints==4);
		TEST_ASSERT(4==list.get_num_owned_points());
		TEST_ASSERT(25==list.get_size_neighborhood_list());
//		cout << "myRank, decomp.sizeNeighborhoodList = " << myRank << ", " << decomp.sizeNeighborhoodList  << endl;

		int myIds[] = {15,1,4,5};
		int n15[] = {1,4,5};
		int n1[] = {15,2,4,5,6};
		int n4[] = {15,1,5,8,9};
		int n5[] = {15,1,2,4,6,8,9,10};
		std::map<int,set<int> > neighMap;
		neighMap[15] = set<int>(n15,n15+3);
		neighMap[1] = set<int>(n1,n1+5);
		neighMap[4] = set<int>(n4,n4+5);
		neighMap[5] = set<int>(n5,n5+8);

		set<int> ids(myIds,myIds+numPoints);
		set<int>::iterator i=ids.begin();
		set<int>::iterator iEnd=ids.end();
		int *gIds = decomp.myGlobalIDs.get();
		int *neigh = list.get_neighborhood().get();
		for(int p=0;p<numPoints;p++,gIds++){
			TEST_ASSERT(iEnd!=ids.find(*gIds));
			set<int>& gIdNeigh = neighMap[*gIds];
			size_t numNeigh = *neigh; neigh++;
//			cout << "myRank, gIds, localId, numNeigh = " << myRank << ", " << *gIds << ", " << p << ", " << numNeigh << endl;
			TEST_ASSERT(gIdNeigh.size() == numNeigh);
			set<int>::iterator neighIter=gIdNeigh.begin();
			set<int>::iterator neighEnd=gIdNeigh.end();
			for(size_t n=0;n<numNeigh;n++,neigh++){
				TEST_ASSERT(neighEnd!=gIdNeigh.find(*neigh));
			}
		}
	}
	if(0==myRank){
		int numPoints = decomp.numPoints;
		TEST_ASSERT(numPoints==4);
		TEST_ASSERT(4==list.get_num_owned_points());
		TEST_ASSERT(25==list.get_size_neighborhood_list());
//		cout << "myRank, decomp.sizeNeighborhoodList = " << myRank << ", " << decomp.sizeNeighborhoodList  << endl;
		int myIds[] = {2,3,6,7};
		int n2[] = {1,3,5,6,7};
		int n3[] = {2,6,7};
		int n6[] = {1,2,3,5,7,9,10,11};
		int n7[] = {2,3,6,10,11};
		std::map<int,set<int> > neighMap;
		neighMap[2] = set<int>(n2,n2+5);
		neighMap[3] = set<int>(n3,n3+3);
		neighMap[6] = set<int>(n6,n6+8);
		neighMap[7] = set<int>(n7,n7+5);

		set<int> ids(myIds,myIds+numPoints);
		set<int>::iterator i=ids.begin();
		set<int>::iterator iEnd=ids.end();
		int *gIds = decomp.myGlobalIDs.get();
		int *neigh = list.get_neighborhood().get();
		for(int p=0;p<numPoints;p++,gIds++){
			TEST_ASSERT(iEnd!=ids.find(*gIds));
			set<int>& gIdNeigh = neighMap[*gIds];
			int numNeigh = *neigh; neigh++;
//			cout << "myRank, gIds, localId, numNeigh = " << myRank << ", " << *gIds << ", " << p << ", " << numNeigh << endl;

			TEST_ASSERT((int)gIdNeigh.size() == numNeigh);
			set<int>::iterator neighIter=gIdNeigh.begin();
			set<int>::iterator neighEnd=gIdNeigh.end();
			for(int n=0;n<numNeigh;n++,neigh++){
				TEST_ASSERT(neighEnd!=gIdNeigh.find(*neigh));
			}
		}


	}
	if(2==myRank){
		int numPoints = decomp.numPoints;
		TEST_ASSERT(numPoints==4);
		TEST_ASSERT(4==list.get_num_owned_points());
		TEST_ASSERT(25==list.get_size_neighborhood_list());
		int myIds[] = {8,9,12,13};
		int n8[] = {4,5,9,12,13};
		int n9[] = {4,5,6,8,10,12,13,14};
		int n12[] = {8,9,13};
		int n13[] = {8,9,10,12,14};
		std::map<int,set<int> > neighMap;
		neighMap[8] = set<int>(n8,n8+5);
		neighMap[9] = set<int>(n9,n9+8);
		neighMap[12] = set<int>(n12,n12+3);
		neighMap[13] = set<int>(n13,n13+5);

		set<int> ids(myIds,myIds+numPoints);
		set<int>::iterator i=ids.begin();
		set<int>::iterator iEnd=ids.end();
		int *gIds = decomp.myGlobalIDs.get();
		int *neigh = list.get_neighborhood().get();
		for(int p=0;p<numPoints;p++,gIds++){
			TEST_ASSERT(iEnd!=ids.find(*gIds));
			set<int>& gIdNeigh = neighMap[*gIds];
			int numNeigh = *neigh; neigh++;
			TEST_ASSERT((int)gIdNeigh.size() == numNeigh);
			set<int>::iterator neighIter=gIdNeigh.begin();
			set<int>::iterator neighEnd=gIdNeigh.end();
			for(int n=0;n<numNeigh;n++,neigh++){
				TEST_ASSERT(neighEnd!=gIdNeigh.find(*neigh));
			}
		}

	}

	if(3==myRank){
		int numPoints = decomp.numPoints;
		TEST_ASSERT(numPoints==4);
		TEST_ASSERT(4==list.get_num_owned_points());
		TEST_ASSERT(25==list.get_size_neighborhood_list());

		int myIds[] = {10,11,14,0};
		int n10[] = {5,6,7,9,11,13,14,0};
		int n11[] = {6,7,10,14,0};
		int n14[] = {13,9,10,11,0};
		int n0[] = {10,11,14};
		std::map<int,set<int> > neighMap;
		neighMap[10] = set<int>(n10,n10+8);
		neighMap[11] = set<int>(n11,n11+5);
		neighMap[14] = set<int>(n14,n14+5);
		neighMap[0] = set<int>(n0,n0+3);
		set<int> ids(myIds,myIds+numPoints);
		set<int>::iterator i=ids.begin();
		set<int>::iterator iEnd=ids.end();
		int *gIds = decomp.myGlobalIDs.get();
		int *neigh = list.get_neighborhood().get();
		for(int p=0;p<numPoints;p++,gIds++){
			TEST_ASSERT(iEnd!=ids.find(*gIds));
			set<int>& gIdNeigh = neighMap[*gIds];
			int numNeigh = *neigh; neigh++;
			TEST_ASSERT((int)gIdNeigh.size() == numNeigh);
			set<int>::iterator neighIter=gIdNeigh.begin();
			set<int>::iterator neighEnd=gIdNeigh.end();
			for(int n=0;n<numNeigh;n++,neigh++){
				TEST_ASSERT(neighEnd!=gIdNeigh.find(*neigh));
			}
		}

	}

}




int main
(
		int argc,
		char* argv[]
)
{

         Teuchos::GlobalMPISession mpiSession(&argc, &argv);
	// Initialize UTF
	return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
}

