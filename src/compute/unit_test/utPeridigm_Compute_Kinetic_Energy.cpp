/*! \file utPeridigm_Kinetic_Energy.cpp */

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

#include <Peridigm_Discretization.hpp>
#include "../Peridigm_Compute_Kinetic_Energy.hpp"
#include "../Peridigm_Compute_Local_Kinetic_Energy.hpp"
#include "../Peridigm_Compute_Global_Kinetic_Energy.hpp"
#include <Peridigm_DataManager.hpp>
#include <Peridigm_DiscretizationFactory.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include "Teuchos_GlobalMPISession.hpp"
#include <Epetra_ConfigDefs.h> // used to define HAVE_MPI
#include <Epetra_Import.h>
#include <vector>
#include "Peridigm.hpp"
#include "Peridigm_Field.hpp"


using namespace Teuchos;
using namespace PeridigmNS;

Teuchos::RCP<Peridigm> createFourPointModel() {
  

  // set up parameter lists
  // these data would normally be read from an input xml file
  Teuchos::RCP<Teuchos::ParameterList> peridigmParams = rcp(new Teuchos::ParameterList());

  // material parameters
  Teuchos::ParameterList& materialParams = peridigmParams->sublist("Materials");
  Teuchos::ParameterList& linearElasticMaterialParams = materialParams.sublist("My Elastic Material");
  linearElasticMaterialParams.set("Material Model", "Elastic");
  linearElasticMaterialParams.set("Density", 7800.0);
  linearElasticMaterialParams.set("Bulk Modulus", 130.0e9);
  linearElasticMaterialParams.set("Shear Modulus", 78.0e9);

  // blocks
  Teuchos::ParameterList& blockParams = peridigmParams->sublist("Blocks");
  Teuchos::ParameterList& blockOneParams = blockParams.sublist("My Group of Blocks");
  blockOneParams.set("Block Names", "block_1");
  blockOneParams.set("Material", "My Elastic Material");
  blockOneParams.set("Horizon", 5.0);

  // Set up discretization parameterlist
  Teuchos::ParameterList& discretizationParams = peridigmParams->sublist("Discretization");
  discretizationParams.set("Type", "PdQuickGrid");

  // pdQuickGrid tensor product mesh generator parameters
  Teuchos::ParameterList& pdQuickGridParams = discretizationParams.sublist("TensorProduct3DMeshGenerator");
  pdQuickGridParams.set("Type", "PdQuickGrid");
  pdQuickGridParams.set("X Origin",  0.0);
  pdQuickGridParams.set("Y Origin",  0.0);
  pdQuickGridParams.set("Z Origin",  0.0);
  pdQuickGridParams.set("X Length",  6.0);
  pdQuickGridParams.set("Y Length",  1.0);
  pdQuickGridParams.set("Z Length",  1.0);
  pdQuickGridParams.set("Number Points X", 4);
  pdQuickGridParams.set("Number Points Y", 1);
  pdQuickGridParams.set("Number Points Z", 1);

  // output parameters (to force instantiation of data storage for compute classes in DataManager)
  Teuchos::ParameterList& outputParams = peridigmParams->sublist("Output");
  Teuchos::ParameterList& outputFields = outputParams.sublist("Output Variables");
  outputFields.set("Kinetic_Energy", true);
  outputFields.set("Global_Kinetic_Energy", true);

  Teuchos::RCP<Discretization> nullDiscretization;
  Teuchos::RCP<Peridigm> peridigm = Teuchos::rcp(new Peridigm(MPI_COMM_WORLD, peridigmParams, nullDiscretization));

  return peridigm;
}

TEUCHOS_UNIT_TEST(Compute_Kinetic_energy, FourPointTest) 
{

  Teuchos::RCP<Epetra_Comm> comm;
  #ifdef HAVE_MPI
    comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  #else
    comm = Teuchos::rcp(new Epetra_SerialComm);
  #endif

  int numProcs = comm->NumProc();

  TEST_COMPARE(numProcs, <=, 4);
  if(numProcs > 4){
    std::cerr << "Unit test runtime ERROR: utPeridigm_Compute_Kinetic_Energy only makes sense on 1 to 4 processors." << std::endl;
    return;
  }

  Teuchos::RCP<Peridigm> peridigm = createFourPointModel();

  FieldManager& fieldManager = FieldManager::self();

  // Get the neighborhood data
  NeighborhoodData neighborhoodData = (*peridigm->getGlobalNeighborhoodData());
  // Access the data we need
  Teuchos::RCP<Epetra_Vector> velocity, volume, kinetic_energy;
  velocity       = peridigm->getBlocks()->begin()->getData(fieldManager.getFieldId("Velocity"), PeridigmField::STEP_NP1);
  volume         = peridigm->getBlocks()->begin()->getData(fieldManager.getFieldId("Volume"), PeridigmField::STEP_NONE);
  kinetic_energy = peridigm->getBlocks()->begin()->getData(fieldManager.getFieldId("Kinetic_Energy"), PeridigmField::STEP_NONE);
  // Get the neighborhood structure
  const int numOwnedPoints = (neighborhoodData.NumOwnedPoints());

  // Manufacture velocity data
  double *velocity_values  = velocity->Values();
  int *myGIDs = velocity->Map().MyGlobalElements();
  int numElements = numOwnedPoints;
  int numTotalElements = volume->Map().NumMyElements();
  for (int i=0;i<numTotalElements;i++) {
    int ID = myGIDs[i];
    velocity_values[3*i] = 3.0*ID;
    velocity_values[3*i+1] = (3.0*ID)+1.0;
    velocity_values[3*i+2] = (3.0*ID)+2.0;
  }

  // Get the blocks
  Teuchos::RCP< std::vector<Block> > blocks = peridigm->getBlocks();

  // Fire the compute classes to fill the kinetic energy data
  peridigm->getComputeManager()->compute(blocks);  

  double density = peridigm->getBlocks()->begin()->getMaterialModel()->Density();
	
  // Now check that volumes and energy is correct
  double *volume_values = volume->Values();
  double *kinetic_energy_values  = kinetic_energy->Values();
  Teuchos::RCP<Epetra_Vector> data = blocks->begin()->getData( fieldManager.getFieldId("Global_Kinetic_Energy"), PeridigmField::STEP_NONE );
  double globalKE = (*data)[0];
  TEST_FLOATING_EQUALITY(globalKE, (double) 2960100, 1.0e-15); 	// Check global scalar value
  for (int i=0;i<numElements;i++)
    TEST_FLOATING_EQUALITY(volume_values[i], 1.5, 1.0e-15);
  for (int i=0;i<numElements;i++) {
    int ID = myGIDs[i];
    double mass = density*volume_values[ID];
    TEST_FLOATING_EQUALITY(kinetic_energy_values[i],  0.5*mass*(pow((3.0*ID),2)+pow(((3.0*ID)+1.0),2)+pow(((3.0*ID)+2.0),2)), 1.0e-15);
    TEST_FLOATING_EQUALITY(globalKE, (double) 2960100, 1.0e-15);
  }
}



int main (int argc, char* argv[]) 
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
}
