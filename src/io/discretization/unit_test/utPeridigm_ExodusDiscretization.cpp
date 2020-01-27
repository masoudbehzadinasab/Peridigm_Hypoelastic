/*! \file utPeridigm_ExodusDiscretization.cpp */

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
#include "Teuchos_GlobalMPISession.hpp"
#include <vector>

#include <Epetra_ConfigDefs.h> // used to define HAVE_MPI
#ifdef HAVE_MPI
  #include <Epetra_MpiComm.h>
#else
  #include <Epetra_SerialComm.h>
#endif
#include "Peridigm_ExodusDiscretization.hpp"
#include "Peridigm_HorizonManager.hpp"

using namespace Teuchos;
using namespace PeridigmNS;

TEUCHOS_UNIT_TEST(ExodusDiscretization, Exodus2x2x2Test) {

  Teuchos::RCP<const Epetra_Comm> comm;
  #ifdef HAVE_MPI
    comm = rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  #else
    comm = rcp(new Epetra_SerialComm);
  #endif
  RCP<ParameterList> discParams = rcp(new ParameterList);

  int numProc = comm->NumProc();
  int myPID = comm->MyPID();
  
  int numMyElementsTruth = 8;
  if(numProc == 2)
    numMyElementsTruth = 4;

  // This test is set up for either 1 or 2 processors
  TEST_ASSERT(numProc == 1 || numProc == 2);

  // create a 2x2x2 discretization
  // specify a neighbor search with the horizon a tad longer than the mesh spacing
  discParams->set("Type", "Exodus");
  discParams->set("Input Mesh File", "utPeridigm_ExodusDiscretization_2x2x2.g");
  discParams->set("Store Exodus Mesh", true);

  // initialize the horizon manager and set the horizon to 0.501
  ParameterList blockParameterList;
  ParameterList& blockParams = blockParameterList.sublist("My Block");
  blockParams.set("Block Names", "block_1");
  blockParams.set("Horizon", 0.501);
  PeridigmNS::HorizonManager::self().loadHorizonInformationFromBlockParameters(blockParameterList);

  // create the discretization
  RCP<ExodusDiscretization> discretization =
    rcp(new ExodusDiscretization(comm, discParams));

  // sanity check, calling with a dimension other than 1 or 3 should throw an exception
  TEST_THROW(discretization->getGlobalOwnedMap(0), Teuchos::Exceptions::InvalidParameter);
  TEST_THROW(discretization->getGlobalOwnedMap(2), Teuchos::Exceptions::InvalidParameter);
  TEST_THROW(discretization->getGlobalOwnedMap(4), Teuchos::Exceptions::InvalidParameter);

  int globalIdOffset = 0;
  if(myPID == 1)
    globalIdOffset = 4;

  // basic checks on the 1d map
  Teuchos::RCP<const Epetra_BlockMap> map = discretization->getGlobalOwnedMap(1);
  TEST_ASSERT(map->NumGlobalElements() == 8);
  TEST_ASSERT(map->NumMyElements() == numMyElementsTruth);
  TEST_ASSERT(map->ElementSize() == 1);
  TEST_ASSERT(map->IndexBase() == 0);
  TEST_ASSERT(map->UniqueGIDs() == true);
  int* myGlobalElements = map->MyGlobalElements();
  for(int i=0 ; i<map->NumMyElements() ; ++i)
      TEST_ASSERT(myGlobalElements[i] == i+globalIdOffset);

  // for the serial case, the map and the overlap map should match
  Teuchos::RCP<const Epetra_BlockMap> overlapMap = discretization->getGlobalOverlapMap(1);
  if(numProc == 1)
    TEST_ASSERT(map->SameAs(*overlapMap) == true);

  // same checks for 3d map
  map = discretization->getGlobalOwnedMap(3);
  TEST_ASSERT(map->NumGlobalElements() == 8);
  TEST_ASSERT(map->NumMyElements() == numMyElementsTruth);
  TEST_ASSERT(map->ElementSize() == 3);
  TEST_ASSERT(map->IndexBase() == 0);
  TEST_ASSERT(map->UniqueGIDs() == true);
  myGlobalElements = map->MyGlobalElements();
  for(int i=0 ; i<map->NumMyElements() ; ++i)
    TEST_ASSERT(myGlobalElements[i] == i+globalIdOffset);

  // for the serial case, the map and the overlap map should match
  overlapMap = discretization->getGlobalOverlapMap(3);
  if(numProc == 1)
    TEST_ASSERT(map->SameAs(*overlapMap) == true);

  // check the bond map
  // the horizon was chosen such that each point should have three neighbors
  Teuchos::RCP<const Epetra_BlockMap> bondMap = discretization->getGlobalBondMap();
  TEST_ASSERT(bondMap->NumGlobalElements() == 8);
  TEST_ASSERT(bondMap->NumMyElements() == numMyElementsTruth);
  TEST_ASSERT(bondMap->IndexBase() == 0);
  TEST_ASSERT(bondMap->UniqueGIDs() == true);
  myGlobalElements = bondMap->MyGlobalElements();
  for(int i=0 ; i<bondMap->NumMyElements() ; ++i)
    TEST_ASSERT(myGlobalElements[i] == i+globalIdOffset);

  TEST_ASSERT(static_cast<int>(discretization->getNumBonds()) == numMyElementsTruth*3);

  // check the initial positions
  // all three coordinates are contained in a single vector
  Teuchos::RCP<Epetra_Vector> initialX = discretization->getInitialX();
  TEST_ASSERT(initialX->MyLength() == numMyElementsTruth*3);
  TEST_ASSERT(initialX->GlobalLength() == 8*3);

  if(myPID == 0){
    TEST_FLOATING_EQUALITY((*initialX)[0],  0.25, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[1],  0.25, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[2],  0.75, 1.0e-15);

    TEST_FLOATING_EQUALITY((*initialX)[3],  0.25, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[4],  0.25, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[5],  0.25, 1.0e-15);
  
    TEST_FLOATING_EQUALITY((*initialX)[6],  0.25, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[7],  0.75, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[8],  0.75, 1.0e-15);

    TEST_FLOATING_EQUALITY((*initialX)[9],  0.25, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[10], 0.75, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[11], 0.25, 1.0e-15);
  }
  if(numProc == 1 || myPID == 2){

    int offset = 0;
    if(numProc == 1)
      offset = 12;

    TEST_FLOATING_EQUALITY((*initialX)[offset+0], 0.75, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[offset+1], 0.25, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[offset+2], 0.75, 1.0e-15);

    TEST_FLOATING_EQUALITY((*initialX)[offset+3], 0.75, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[offset+4], 0.25, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[offset+5], 0.25, 1.0e-15);

    TEST_FLOATING_EQUALITY((*initialX)[offset+6], 0.75, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[offset+7], 0.75, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[offset+8], 0.75, 1.0e-15);

    TEST_FLOATING_EQUALITY((*initialX)[offset+9], 0.75, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[offset+10], 0.75, 1.0e-15);
    TEST_FLOATING_EQUALITY((*initialX)[offset+11], 0.25, 1.0e-15);
  }

  // check cell volumes
  Teuchos::RCP<Epetra_Vector> volume = discretization->getCellVolume();
  TEST_ASSERT(volume->MyLength() == numMyElementsTruth);
  TEST_ASSERT(volume->GlobalLength() == 8);
  for(int i=0 ; i<volume->MyLength() ; ++i)
    TEST_FLOATING_EQUALITY((*volume)[i], 0.125, 1.0e-15);

  // check the neighbor lists
  Teuchos::RCP<PeridigmNS::NeighborhoodData> neighborhoodData = discretization->getNeighborhoodData();
  TEST_ASSERT(neighborhoodData->NumOwnedPoints() == numMyElementsTruth);
  int* ownedIds = neighborhoodData->OwnedIDs();
  for(int i=0 ; i<neighborhoodData->NumOwnedPoints() ; ++i)
    TEST_ASSERT(ownedIds[i] == i);
  TEST_ASSERT(neighborhoodData->NeighborhoodListSize() == numMyElementsTruth*4);
  int* neighborhood = neighborhoodData->NeighborhoodList();
  int* neighborhoodPtr = neighborhoodData->NeighborhoodPtr();

  if(numProc == 1){

    // The global and local IDs are the same in the serial case

    TEST_ASSERT(neighborhoodPtr[0] == 0);
    TEST_ASSERT(neighborhood[0]    == 3);
    TEST_ASSERT(neighborhood[1]    == 1);
    TEST_ASSERT(neighborhood[2]    == 2);
    TEST_ASSERT(neighborhood[3]    == 4);

    TEST_ASSERT(neighborhoodPtr[1] == 4);
    TEST_ASSERT(neighborhood[4]    == 3);
    TEST_ASSERT(neighborhood[5]    == 0);
    TEST_ASSERT(neighborhood[6]    == 3);
    TEST_ASSERT(neighborhood[7]    == 5);

    TEST_ASSERT(neighborhoodPtr[2] == 8);
    TEST_ASSERT(neighborhood[8]    == 3);
    TEST_ASSERT(neighborhood[9]    == 0);
    TEST_ASSERT(neighborhood[10]   == 3);
    TEST_ASSERT(neighborhood[11]   == 6);

    TEST_ASSERT(neighborhoodPtr[3] == 12);
    TEST_ASSERT(neighborhood[12]   == 3);
    TEST_ASSERT(neighborhood[13]   == 1);
    TEST_ASSERT(neighborhood[14]   == 2);
    TEST_ASSERT(neighborhood[15]   == 7);

    TEST_ASSERT(neighborhoodPtr[4] == 16);
    TEST_ASSERT(neighborhood[16]   == 3);
    TEST_ASSERT(neighborhood[17]   == 0);
    TEST_ASSERT(neighborhood[18]   == 5);
    TEST_ASSERT(neighborhood[19]   == 6);

    TEST_ASSERT(neighborhoodPtr[5] == 20);
    TEST_ASSERT(neighborhood[20]   == 3);
    TEST_ASSERT(neighborhood[21]   == 1);
    TEST_ASSERT(neighborhood[22]   == 4);
    TEST_ASSERT(neighborhood[23]   == 7);

    TEST_ASSERT(neighborhoodPtr[6] == 24);
    TEST_ASSERT(neighborhood[24]   == 3);
    TEST_ASSERT(neighborhood[25]   == 2);
    TEST_ASSERT(neighborhood[26]   == 4);
    TEST_ASSERT(neighborhood[27]   == 7);

    TEST_ASSERT(neighborhoodPtr[7] == 28);
    TEST_ASSERT(neighborhood[28]   == 3);
    TEST_ASSERT(neighborhood[29]   == 3);
    TEST_ASSERT(neighborhood[30]   == 5);
    TEST_ASSERT(neighborhood[31]   == 6);
  }
  if(numProc == 2 && myPID == 0){

    overlapMap = discretization->getGlobalOverlapMap(1);

    TEST_ASSERT(neighborhoodPtr[0] == 0);
    TEST_ASSERT(neighborhood[0]    == 3);
    TEST_ASSERT(neighborhood[1]    == overlapMap->LID(1));
    TEST_ASSERT(neighborhood[2]    == overlapMap->LID(2));
    TEST_ASSERT(neighborhood[3]    == overlapMap->LID(4));

    TEST_ASSERT(neighborhoodPtr[1] == 4);
    TEST_ASSERT(neighborhood[4]    == 3);
    TEST_ASSERT(neighborhood[5]    == overlapMap->LID(0));
    TEST_ASSERT(neighborhood[6]    == overlapMap->LID(3));
    TEST_ASSERT(neighborhood[7]    == overlapMap->LID(5));

    TEST_ASSERT(neighborhoodPtr[2] == 8);
    TEST_ASSERT(neighborhood[8]    == 3);
    TEST_ASSERT(neighborhood[9]    == overlapMap->LID(0));
    TEST_ASSERT(neighborhood[10]   == overlapMap->LID(3));
    TEST_ASSERT(neighborhood[11]   == overlapMap->LID(6));

    TEST_ASSERT(neighborhoodPtr[3] == 12);
    TEST_ASSERT(neighborhood[12]   == 3);
    TEST_ASSERT(neighborhood[13]   == overlapMap->LID(1));
    TEST_ASSERT(neighborhood[14]   == overlapMap->LID(2));
    TEST_ASSERT(neighborhood[15]   == overlapMap->LID(7));
  }
  if(numProc == 2 && myPID == 1){

    overlapMap = discretization->getGlobalOverlapMap(1);

    // Note, the neighborlists get rearranged on processor #2 relative to the serial case

    TEST_ASSERT(neighborhoodPtr[0] == 0);
    TEST_ASSERT(neighborhood[0]    == 3);
    TEST_ASSERT(neighborhood[1]    == overlapMap->LID(5));
    TEST_ASSERT(neighborhood[2]    == overlapMap->LID(6));
    TEST_ASSERT(neighborhood[3]    == overlapMap->LID(0));

    TEST_ASSERT(neighborhoodPtr[1] == 4);
    TEST_ASSERT(neighborhood[4]    == 3);
    TEST_ASSERT(neighborhood[5]    == overlapMap->LID(4));
    TEST_ASSERT(neighborhood[6]    == overlapMap->LID(7));
    TEST_ASSERT(neighborhood[7]    == overlapMap->LID(1));

    TEST_ASSERT(neighborhoodPtr[2] == 8);
    TEST_ASSERT(neighborhood[8]    == 3);
    TEST_ASSERT(neighborhood[9]    == overlapMap->LID(4));
    TEST_ASSERT(neighborhood[10]   == overlapMap->LID(7));
    TEST_ASSERT(neighborhood[11]   == overlapMap->LID(2));

    TEST_ASSERT(neighborhoodPtr[3] == 12);
    TEST_ASSERT(neighborhood[12]   == 3);
    TEST_ASSERT(neighborhood[13]   == overlapMap->LID(5));
    TEST_ASSERT(neighborhood[14]   == overlapMap->LID(6));
    TEST_ASSERT(neighborhood[15]   == overlapMap->LID(3));
  }

  // check the positions of the nodes in the original Exodus II mesh file
  std::vector<double> exodusNodePositions;

  discretization->getExodusMeshNodePositions(0, exodusNodePositions);
  TEST_ASSERT(exodusNodePositions.size() == 8*3);
  TEST_FLOATING_EQUALITY(exodusNodePositions[0],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[1],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[2],  1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[3],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[4],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[5],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[6],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[7],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[8],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[9],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[10], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[11], 1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[12], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[13], 0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[14], 1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[15], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[16], 0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[17], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[18], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[19], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[20], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[21], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[22], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[23], 1.0, 1.0e-16);

  discretization->getExodusMeshNodePositions(1, exodusNodePositions);
  TEST_ASSERT(exodusNodePositions.size() == 8*3);
  TEST_FLOATING_EQUALITY(exodusNodePositions[0],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[1],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[2],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[3],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[4],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[5],  0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[6],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[7],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[8],  0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[9],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[10], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[11], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[12], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[13], 0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[14], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[15], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[16], 0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[17], 0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[18], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[19], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[20], 0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[21], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[22], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[23], 0.5, 1.0e-16);

  discretization->getExodusMeshNodePositions(2, exodusNodePositions);
  TEST_ASSERT(exodusNodePositions.size() == 8*3);
  TEST_FLOATING_EQUALITY(exodusNodePositions[0],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[1],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[2],  1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[3],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[4],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[5],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[6],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[7],  1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[8],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[9],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[10], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[11], 1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[12], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[13], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[14], 1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[15], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[16], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[17], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[18], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[19], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[20], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[21], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[22], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[23], 1.0, 1.0e-16);

  discretization->getExodusMeshNodePositions(3, exodusNodePositions);
  TEST_ASSERT(exodusNodePositions.size() == 8*3);
  TEST_FLOATING_EQUALITY(exodusNodePositions[0],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[1],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[2],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[3],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[4],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[5],  0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[6],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[7],  1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[8],  0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[9],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[10], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[11], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[12], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[13], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[14], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[15], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[16], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[17], 0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[18], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[19], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[20], 0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[21], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[22], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[23], 0.5, 1.0e-16);

  discretization->getExodusMeshNodePositions(4, exodusNodePositions);
  TEST_ASSERT(exodusNodePositions.size() == 8*3);
  TEST_FLOATING_EQUALITY(exodusNodePositions[0],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[1],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[2],  1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[3],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[4],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[5],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[6],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[7],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[8],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[9],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[10], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[11], 1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[12], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[13], 0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[14], 1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[15], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[16], 0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[17], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[18], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[19], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[20], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[21], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[22], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[23], 1.0, 1.0e-16);

  discretization->getExodusMeshNodePositions(5, exodusNodePositions);
  TEST_ASSERT(exodusNodePositions.size() == 8*3);
  TEST_FLOATING_EQUALITY(exodusNodePositions[0],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[1],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[2],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[3],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[4],  0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[5],  0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[6],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[7],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[8],  0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[9],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[10], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[11], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[12], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[13], 0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[14], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[15], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[16], 0.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[17], 0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[18], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[19], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[20], 0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[21], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[22], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[23], 0.5, 1.0e-16);

  discretization->getExodusMeshNodePositions(6, exodusNodePositions);
  TEST_ASSERT(exodusNodePositions.size() == 8*3);
  TEST_FLOATING_EQUALITY(exodusNodePositions[0],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[1],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[2],  1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[3],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[4],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[5],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[6],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[7],  1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[8],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[9],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[10], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[11], 1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[12], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[13], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[14], 1.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[15], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[16], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[17], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[18], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[19], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[20], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[21], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[22], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[23], 1.0, 1.0e-16);

  discretization->getExodusMeshNodePositions(7, exodusNodePositions);
  TEST_ASSERT(exodusNodePositions.size() == 8*3);
  TEST_FLOATING_EQUALITY(exodusNodePositions[0],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[1],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[2],  0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[3],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[4],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[5],  0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[6],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[7],  1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[8],  0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[9],  0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[10], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[11], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[12], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[13], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[14], 0.5, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[15], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[16], 0.5, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[17], 0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[18], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[19], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[20], 0.0, 1.0e-16);

  TEST_FLOATING_EQUALITY(exodusNodePositions[21], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[22], 1.0, 1.0e-16);
  TEST_FLOATING_EQUALITY(exodusNodePositions[23], 0.5, 1.0e-16);    
}

int main
(int argc, char* argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
}
