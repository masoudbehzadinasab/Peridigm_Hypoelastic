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

#include <boost/property_tree/json_parser.hpp>
#include <string>
#include "QuickGrid.h"
#include "QuickGridData.h"
#include "calculators.h"
#include "NeighborhoodList.h"
#include "PdZoltan.h"
#include "PdutMpiFixture.h"
#include "Field.h"
#include "Vector3D.h"
#include "Array.h"
#include <set>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include "Epetra_Comm.h"
#include "Epetra_BlockMap.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"

#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#include "mpi.h"
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include <iostream>
#include <cmath>
using namespace Pdut;

using namespace QUICKGRID;
using UTILITIES::Vector3D;
using UTILITIES::Array;
using Field_NS::Field;
using Field_NS::FieldSpec;
using std::tr1::shared_ptr;

using std::size_t;
using std::string;
using std::cout;

static size_t myRank;
static size_t numProcs;
const string json_filename="./input_files/ut_CartesianTensorProductVolumeFraction.json";

void compute_neighborhood_volumes
(
		const PDNEIGH::NeighborhoodList& list,
		Field<double>& neighborhoodVol,
		Field<double>& naiveNeighborhoodVol,
		Array<double>& overlapCellVol,
		shared_ptr<double> xOverlapPtr,
		const BOND_VOLUME::QUICKGRID::VolumeFractionCalculator& calculator
)
{
	size_t N = list.get_num_owned_points();
	

	const int *neighPtr = list.get_local_neighborhood().get();
	const double *xOwned = list.get_owned_x().get();
	const double *xOverlap = xOverlapPtr.get();
	const double *cellVolOverlap = overlapCellVol.get();
	double *neighVol = neighborhoodVol.get();
	double *naiveNeighVol = naiveNeighborhoodVol.get();
	for(size_t p=0;p<N;p++, xOwned +=3, neighVol++,naiveNeighVol++){
		int numNeigh = *neighPtr; neighPtr++;
		/*
		 * initialize neighborhood to zero;
		 * computes contributions from neighborhood and does not
		 * include self volume
		 */
		*neighVol = 0.0;
		*naiveNeighVol = 0.0;

		const double *P = xOwned;

		/*
		 * Loop over neighborhood of point P and compute
		 * fractional volume
		 */
		for(int n=0;n<numNeigh;n++,neighPtr++){
			int localId = *neighPtr;
			const double *Q = &xOverlap[3*localId];
			double cellVolume = cellVolOverlap[localId];
			*neighVol += calculator(P,Q);
			*naiveNeighVol += cellVolume;
		}
	}
}

void compute_cell_volumes
(
		const PDNEIGH::NeighborhoodList& list,
		Field<double>& specialCellVolume,
		shared_ptr<double> xOverlapPtr,
		const BOND_VOLUME::QUICKGRID::VolumeFractionCalculator& calculator
)
{
	size_t N = list.get_num_owned_points();
	
	const double *xOwned = list.get_owned_x().get();
	double *vOwned = specialCellVolume.get();
	for(size_t p=0;p<N;p++, xOwned +=3, vOwned++){

		const double *P = xOwned;

		/*
		 * compute cell volume using quick grid quadrature
		 */
		*vOwned=calculator.cellVolume(P);

	}
}


TEUCHOS_UNIT_TEST(VolumeFraction, CubeTest) {

    // Create an empty property tree object
    using boost::property_tree::ptree;
    ptree pt;

    // Load the json file into the property tree. If reading fails
    // (cannot open file, parse error), an exception is thrown.
    read_json(json_filename, pt);

    /*
     * Get Discretization
     */
    ptree discretization_tree=pt.find("Discretization")->second;
    string path=discretization_tree.get<string>("Type");
    double horizon=pt.get<double>("Discretization.Horizon");

	double xStart = pt.get<double>(path+".X Origin");
	TEST_ASSERT(0.0==xStart);
	double yStart = pt.get<double>(path+".Y Origin");
	TEST_ASSERT(0.0==yStart);
	double zStart = pt.get<double>(path+".Z Origin");
	TEST_ASSERT(0.0==zStart);

	double xLength = pt.get<double>(path+".X Length");
	TEST_ASSERT(1.0==xLength);
	double yLength = pt.get<double>(path+".Y Length");
	TEST_ASSERT(1.0==yLength);
	double zLength = pt.get<double>(path+".Z Length");
	TEST_ASSERT(1.0==zLength);


	const int nx = pt.get<int>(path+".Number Points X");
	TEST_ASSERT(10==nx);
	const int ny = pt.get<int>(path+".Number Points Y");
	TEST_ASSERT(10==ny);
	const int nz = pt.get<int>(path+".Number Points Z");
	TEST_ASSERT(10==nz);
	const QUICKGRID::Spec1D xSpec(nx,xStart,xLength);
	const QUICKGRID::Spec1D ySpec(ny,yStart,yLength);
	const QUICKGRID::Spec1D zSpec(nz,zStart,zLength);

	BOND_VOLUME::QUICKGRID::VolumeFractionCalculator calculator(xSpec,ySpec,zSpec,horizon);

	// Create decomposition iterator
	QUICKGRID::TensorProduct3DMeshGenerator cellPerProcIter(numProcs, horizon, xSpec, ySpec, zSpec);
	QUICKGRID::QuickGridData gridData = QUICKGRID::getDiscretization(myRank, cellPerProcIter);;


	// This load-balances
	gridData = PDNEIGH::getLoadBalancedDiscretization(gridData);

	/*
	 * Create neighborhood with an enlarged horizon
	 */
	double cell_diagonal=calculator.get_cell_diagonal();
	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	PDNEIGH::NeighborhoodList list(comm,gridData.zoltanPtr.get(),gridData.numPoints,gridData.myGlobalIDs,gridData.myX,horizon+cell_diagonal);
	Array<double> xOverlapArray;
	Array<double> vOverlapArray;
	{
		/*
		 * mesh coordinates overlap
		 */
		Epetra_BlockMap ownedMap(*list.getOwnedMap(3));
		Epetra_BlockMap overlapMap(*list.getOverlapMap(3));
		xOverlapArray = Array<double>(overlapMap.NumMyElements()*3);
		Epetra_Import importNDF(overlapMap,ownedMap);
		Epetra_Vector xOverlap(View,overlapMap,xOverlapArray.get());
		Epetra_Vector xOwned(View,ownedMap,list.get_owned_x().get());
		xOverlap.Import(xOwned,importNDF,Insert);
	}

	{
		/*
		 * volume overlap
		 */
		Epetra_BlockMap ownedMap(*list.getOwnedMap(1));
		Epetra_BlockMap overlapMap(*list.getOverlapMap(1));
		vOverlapArray = Array<double>(overlapMap.NumMyElements()*1);
		vOverlapArray.set(0.0);
		Epetra_Import importNDF(overlapMap,ownedMap);
		Epetra_Vector vOverlap(View,overlapMap,vOverlapArray.get());
		Epetra_Vector vOwned(View,ownedMap,gridData.cellVolume.get());
		vOverlap.Import(vOwned,importNDF,Insert);
	}

	/*
	 * Compute volume on neighborhood for every point in mesh;
	 * Points with a spherical neighborhood that are completely enclosed
	 * should have a volume that very closely matches the analytical value for a sphere
	 */
	FieldSpec neighVolSpec
	(Field_ENUM::TYPE_UNDEFINED,Field_ENUM::ELEMENT,Field_ENUM::SCALAR, Field_ENUM::CONSTANT,"neighVol");

	FieldSpec naiveNeighVolSpec
	(Field_ENUM::TYPE_UNDEFINED,Field_ENUM::ELEMENT,Field_ENUM::SCALAR,Field_ENUM::CONSTANT,"naiveNeighVol");

	FieldSpec quadratureCellVolSpec
	(Field_ENUM::TYPE_UNDEFINED,Field_ENUM::ELEMENT,Field_ENUM::SCALAR,Field_ENUM::CONSTANT,"quadratureCellVol");

	Field<double> neighVol(neighVolSpec,list.get_num_owned_points());
	Field<double> naiveNeighVol(naiveNeighVolSpec,list.get_num_owned_points());
	Field<double> quadratureCellVol(quadratureCellVolSpec,list.get_num_owned_points());
	Field<double> cellVol(Field_NS::VOLUME,gridData.cellVolume,list.get_num_owned_points());

        size_t N = list.get_num_owned_points();


   
	TEST_ASSERT(quadratureCellVol.get_num_points()==N);

	compute_cell_volumes(list,quadratureCellVol,xOverlapArray.get_shared_ptr(),calculator);

        
	TEST_ASSERT(neighVol.get_num_points()==N);
	TEST_ASSERT(naiveNeighVol.get_num_points()==N);

        
	compute_neighborhood_volumes(list,neighVol,naiveNeighVol,vOverlapArray,xOverlapArray.get_shared_ptr(),calculator);
}






int main
(
		int argc,
		char* argv[]
)
{
	// Initialize MPI and timer
	PdutMpiFixture myMpi = PdutMpiFixture(argc,argv);

	// These are static (file scope) variables
	myRank = myMpi.rank;
	numProcs = myMpi.numProcs;

	// Initialize UTF
	return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
}

