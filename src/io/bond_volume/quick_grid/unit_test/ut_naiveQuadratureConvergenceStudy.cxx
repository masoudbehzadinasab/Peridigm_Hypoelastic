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

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Field.h"
#include "QuickGrid.h"
#include "calculators.h"
#include "material_utilities.h"
#include "NeighborhoodList.h"
#include "PdZoltan.h"
#include "BondFilter.h"
#include "Array.h"
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#include "mpi.h"
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

using std::size_t;
using std::tr1::shared_ptr;
using UTILITIES::Array;
using UTILITIES::Vector3D;
using std::pair;
using std::cout;
using std::endl;

static int nx;
static int ny;
static int nz;
static double horizon;
const size_t numProcs=1;
const size_t myRank=0;

void probe_shear
(
		MATERIAL_EVALUATION::PURE_SHEAR mode,
		Array<int> neighborhoodPtr,
		Array<double> X,
		Array<double> xPtr,
		Array<double> Y,
		Array<double> yPtr,
		Array<double> bondVolume,
		double horizon,
		double gamma,
		double m_code
);


void set_static_data(const std::string& json_filename)
{
	// Create an empty property tree object
	using boost::property_tree::ptree;
	ptree pt;

	try {
		read_json(json_filename, pt);
	} catch(std::exception& e){
		std::cerr << e.what();
		std::exit(1);
	}

	/*
	 * Get Discretization
	 */
	ptree discretization_tree=pt.find("Discretization")->second;
	std::string path=discretization_tree.get<std::string>("Type");
	horizon=pt.get<double>("Discretization.Horizon");

	if("QuickGrid.TensorProduct3DMeshGenerator"==path){

		nx = pt.get<int>(path+".Number Points X");
		ny = pt.get<int>(path+".Number Points Y");
		nz = pt.get<int>(path+".Number Points Z");

	} else {
		std::string s;
		s = "Error-->ut_naiveQuadratureConvergenceStudy\n";
		s += "\tTest only works for Discretization.Type==QuickGrid.TensorProduct3DMeshGenerator\n";
		throw std::runtime_error(s);
	}

}


void write_table_1_header(const std::string& output_tex_table){
	std::stringstream table_out;

	table_out << "\\begin{table}[ht]" << "\n";
	table_out << "\\centering" << "\n";
	table_out << "\\bigskip" << "\n";
	table_out << "\\begin{tabular}{|c|c|c|c|}" << "\n";
	table_out << "\\hline" << "\n";
	table_out << "$n$ "
			    << "& $\\frac{|m-m_n|}{m}$ "
			    << "& $\\frac{\\Vert e^d\\Vert^2-\\Vert e^d_n\\Vert^2}{\\Vert e^d\\Vert^2}$ "
			    << "& $\\frac{\\Vert e^d\\Vert^2}{\\Vert e^d_n\\Vert^2}$ \\\\" << "\n";
	table_out << "\\hline" << "\n";


	std::ofstream file_stream;
	file_stream.open(output_tex_table.c_str(),std::ios::app|std::ios::out);

	file_stream << table_out.str();
	file_stream.close();

}

void close_table_1(const std::string& output_tex_table) {
	std::stringstream table_out;

	table_out << "\\hline" << "\n";
	table_out << "\\end{tabular}" << "\n";
	table_out << "\\end{table}" << "\n";
	std::ofstream file_stream;
	file_stream.open(output_tex_table.c_str(),std::ios::app|std::ios::out);
	file_stream << table_out.str();
	file_stream.close();
}


QUICKGRID::QuickGridData getGrid(const string& _json_filename) {
	shared_ptr<QUICKGRID::QuickGridMeshGenerationIterator> g;
	g = QUICKGRID::getMeshGenerator(numProcs,_json_filename);
	QUICKGRID::QuickGridData decomp =  QUICKGRID::getDiscretization(myRank, *g);

	// This load-balances
	decomp = PDNEIGH::getLoadBalancedDiscretization(decomp);
	return decomp;
}

void probe_shear
(
	MATERIAL_EVALUATION::PURE_SHEAR mode,
	Array<int> neighborhoodPtr,
	Array<double> X,
	Array<double> xPtr,
	Array<double> Y,
	Array<double> yPtr,
	Array<double> cellVolume,
	double horizon,
	double gamma,
	double m_code
)
{

	/*
	 * This is the reference value for the weighted volume
	 */
	double m_analytical = 4.0 * M_PI * pow(horizon,5) / 5.0;
	double m_err = std::fabs(m_analytical-m_code)/m_analytical;
	/*
	 * NOTE: X is center of sphere and there no displacement at this point
	 * therefore, Y=X
	 */
	/*MATERIAL_EVALUATION::set_pure_shear(neighborhoodPtr.get(),X.get(),xPtr.get(),yPtr.get(),mode,gamma);
	
       
        double theta = MATERIAL_EVALUATION::computeDilatation(neighborhoodPtr.get(),X.get(),xPtr.get(),X.get(),yPtr.get(),cellVolume.get(),m_code,horizon);
	std::cout << "ut_naiveQuadratureConvergenceStudy::probe_shear dilatation = " << theta << std::endl;
	double tolerance=1.0e-12;
	BOOST_CHECK_SMALL(theta,tolerance);
*/
      

	/*
	 * compute shear correction factor
	 */
	/*
	 * This is the reference value for ed_squared
	 */
	double reference = 4.0 * M_PI * gamma * gamma * pow(horizon,5) / 75.0;
	double ed2 = MATERIAL_EVALUATION::compute_norm_2_deviatoric_extension(neighborhoodPtr.get(),X.get(),xPtr.get(),Y.get(),yPtr.get(),cellVolume.get(),m_code,horizon);
	double scf = reference/ed2;
	double ed_err = fabs(reference-ed2)/reference;
	std::cout << "ut_scf::probe_shear MODE = " << mode << std::endl;
	std::cout << "ut_scf::ed^2 = " << reference << std::endl;
	cout.precision(2);
	std::cout << std::scientific << "ut_scf::probe_shear computed % ed_err in pure shear = " << 100*ed_err << std::endl;
	std::cout << "ut_scf::probe_shear computed scf in pure shear = " << scf << std::endl;

	std::stringstream table_1_out;
	table_1_out << nx << " & ";
	table_1_out.precision(4);
	table_1_out << m_err*100 << "\\% & ";
	table_1_out << ed_err*100 << "\\% & ";
	table_1_out.precision(3);
	table_1_out << scf << " \\\\ \n";

	/*
	 * write latex table
	 */
	std::ofstream file_stream;
	file_stream.open("naive_table_1.tex",std::ios::app|std::ios::out);
	file_stream << table_1_out.str();
	file_stream.close();

	/*
	 * write raw data
	 */
	file_stream.open("ut_naiveQuadratureConvergenceStudy.dat",std::ios::app|std::ios::out);
	file_stream << nx << " ";
	file_stream << std::scientific;
	file_stream.precision(12);
	file_stream << 2.0*horizon/nx << " ";
	file_stream << m_code << " ";
	file_stream << ed2 << "\n";
	file_stream.close();

}

void scf_probe(PDNEIGH::NeighborhoodList list, Array<int> &neighborhoodPtr, Array<double> xPtr, Array<double> &X, Array<double> &Y, Array<double> cellVolume, Array<double> &yPtr, double gamma, double &m_analytical, double &m_code, double &rel_diff, double &theta, size_t gId, size_t num_neigh ){

   {
		/*
		 * copy neighborhood list for center point over to array
		 */
		const int *neighborhood = list.get_neighborhood(gId);
		
		for(size_t j=0;j<num_neigh+1;j++,neighborhood++)
			neighborhoodPtr[j]=*neighborhood;
	}

        /*
	 * X is the center of the sphere
	 */
	 X.set(0.0);
	/*
	 * Y = X since we are fixing the center of the sphere
	 */
	Y.set(0.0);

        m_analytical = 4.0 * M_PI * pow(horizon,5) / 5.0;
	m_code = MATERIAL_EVALUATION::computeWeightedVolume(X.get(),xPtr.get(),cellVolume.get(),neighborhoodPtr.get(),horizon);
	rel_diff = std::abs(m_analytical-m_code)/m_analytical;
	std::cout << std::scientific;
	std::cout.precision(3);
	std::cout << "ut_scf::analytical value for weighted volume on sphere = " << m_analytical << std::endl;
	std::cout << "ut_scf::code computed weighted volume on sphere = " << m_code << std::endl;
	std::cout << "ut_scf::% relative error weighted volume = " << 100*rel_diff << std::endl;

	

       MATERIAL_EVALUATION::set_pure_shear(neighborhoodPtr.get(),X.get(),xPtr.get(),yPtr.get(),MATERIAL_EVALUATION::XY,gamma);
       theta = MATERIAL_EVALUATION::computeDilatation(neighborhoodPtr.get(),X.get(),xPtr.get(),X.get(),yPtr.get(),cellVolume.get(),m_code,horizon);
       std::cout << "ut_naiveQuadratureConvergenceStudy::probe_shear dilatation = " << theta << std::endl;

}





TEUCHOS_UNIT_TEST(NaiveQuadratureConvergenceStudy, n3Test) {

	std::string json_filename = "./input_files/ut_bondVolumeConvergenceStudy_n=3.json";
	set_static_data(json_filename);
         /*
	 * Unit test looks exclusively at the cell at the center of cube;
	 * This cell ID depends upon nx, ny, nz
	 *
	 * MESH INPUT MUST HAVE EVEN NUMBER OF CELLS
	 */
	TEST_ASSERT(0==(nx+1)%2);
	/*
	 * mesh must have equal number of cells along each axis
	 */
	TEST_ASSERT(nx==ny);
	TEST_ASSERT(nx==nz);

	QUICKGRID::QuickGridData gridData = getGrid(json_filename);

	// This load-balances
	gridData = PDNEIGH::getLoadBalancedDiscretization(gridData);

	/*
	 * Create neighborhood with an enlarged horizon
	 */

	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	PDNEIGH::NeighborhoodList list(comm,gridData.zoltanPtr.get(),gridData.numPoints,gridData.myGlobalIDs,gridData.myX,horizon);


	// coordinate indices of center cell
	size_t ic = (nx -1)/2;
	size_t jc = ic;
	size_t kc = ic;
	size_t gId = nx * ny * kc + nx * jc + ic;

	/**
	 * WARNING: note following ASSUMPTION -- gId == local id
	 * CAUTION: this test only makes sense in 'serial' -- local id
	 * and gId are not the same in parallel.
	 */
	size_t num_neigh = list.get_num_neigh(gId);

	Array<int> neighborhoodPtr(1+num_neigh);
        Array<double> xPtr(list.get_num_owned_points()*3,list.get_owned_x());
        Array<double> X(3);
        Array<double> Y(3); 
        Array<double> cellVolume(gridData.numPoints,gridData.cellVolume);
        Array<double> yPtr(3*list.get_num_owned_points());
        double gamma = 1.0e-6;
        

        const int *neighborhood = list.get_neighborhood(gId);
	TEST_ASSERT((int)num_neigh == *neighborhood);
	
	/*
	 * expectation is that cell center is at origin
	 */

	TEST_COMPARE(xPtr[3*gId+0], <=,1.0e-15);
	TEST_COMPARE(xPtr[3*gId+1], <=, 1.0e-15);
	TEST_COMPARE(xPtr[3*gId+2], <=, 1.0e-15);

        double m_analytical, m_code, rel_diff, theta;
	
	scf_probe(list, neighborhoodPtr, xPtr, X, Y, cellVolume, yPtr, gamma, m_analytical, m_code, rel_diff, theta, gId, num_neigh );

	
       double tolerance=1.0e-12;
       TEST_COMPARE(theta, <=, tolerance);
       
	/*
	 * PROBE XY
	 */
      probe_shear(MATERIAL_EVALUATION::XY,neighborhoodPtr,X,xPtr,Y,yPtr,cellVolume,horizon,gamma,m_code);
	
}


TEUCHOS_UNIT_TEST(NaiveQuadratureConvergenceStudy, n5Test) {

	std::string json_filename = "./input_files/ut_bondVolumeConvergenceStudy_n=5.json";
	set_static_data(json_filename);
         /*
	 * Unit test looks exclusively at the cell at the center of cube;
	 * This cell ID depends upon nx, ny, nz
	 *
	 * MESH INPUT MUST HAVE EVEN NUMBER OF CELLS
	 */
	TEST_ASSERT(0==(nx+1)%2);
	/*
	 * mesh must have equal number of cells along each axis
	 */
	TEST_ASSERT(nx==ny);
	TEST_ASSERT(nx==nz);

	QUICKGRID::QuickGridData gridData = getGrid(json_filename);

	// This load-balances
	gridData = PDNEIGH::getLoadBalancedDiscretization(gridData);

	/*
	 * Create neighborhood with an enlarged horizon
	 */

	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	PDNEIGH::NeighborhoodList list(comm,gridData.zoltanPtr.get(),gridData.numPoints,gridData.myGlobalIDs,gridData.myX,horizon);


	// coordinate indices of center cell
	size_t ic = (nx -1)/2;
	size_t jc = ic;
	size_t kc = ic;
	size_t gId = nx * ny * kc + nx * jc + ic;

	/**
	 * WARNING: note following ASSUMPTION -- gId == local id
	 * CAUTION: this test only makes sense in 'serial' -- local id
	 * and gId are not the same in parallel.
	 */
	size_t num_neigh = list.get_num_neigh(gId);

	Array<int> neighborhoodPtr(1+num_neigh);
        Array<double> xPtr(list.get_num_owned_points()*3,list.get_owned_x());
        Array<double> X(3);
        Array<double> Y(3); 
        Array<double> cellVolume(gridData.numPoints,gridData.cellVolume);
        Array<double> yPtr(3*list.get_num_owned_points());
        double gamma = 1.0e-6;
        

        const int *neighborhood = list.get_neighborhood(gId);
	TEST_ASSERT((int)num_neigh == *neighborhood);
	
	/*
	 * expectation is that cell center is at origin
	 */

	TEST_COMPARE(xPtr[3*gId+0], <=,1.0e-15);
	TEST_COMPARE(xPtr[3*gId+1], <=, 1.0e-15);
	TEST_COMPARE(xPtr[3*gId+2], <=, 1.0e-15);

        double m_analytical, m_code, rel_diff, theta;
	
	scf_probe(list, neighborhoodPtr, xPtr, X, Y, cellVolume, yPtr, gamma, m_analytical, m_code, rel_diff, theta, gId, num_neigh );

	
       double tolerance=1.0e-12;
       TEST_COMPARE(theta, <=, tolerance);
       
	/*
	 * PROBE XY
	 */
      probe_shear(MATERIAL_EVALUATION::XY,neighborhoodPtr,X,xPtr,Y,yPtr,cellVolume,horizon,gamma,m_code);
	
	
}

TEUCHOS_UNIT_TEST(NaiveQuadratureConvergenceStudy, n7Test) {

	std::string json_filename = "./input_files/ut_bondVolumeConvergenceStudy_n=7.json";
	set_static_data(json_filename);
        /*
	 * Unit test looks exclusively at the cell at the center of cube;
	 * This cell ID depends upon nx, ny, nz
	 *
	 * MESH INPUT MUST HAVE EVEN NUMBER OF CELLS
	 */
	TEST_ASSERT(0==(nx+1)%2);
	/*
	 * mesh must have equal number of cells along each axis
	 */
	TEST_ASSERT(nx==ny);
	TEST_ASSERT(nx==nz);

	QUICKGRID::QuickGridData gridData = getGrid(json_filename);

	// This load-balances
	gridData = PDNEIGH::getLoadBalancedDiscretization(gridData);

	/*
	 * Create neighborhood with an enlarged horizon
	 */

	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	PDNEIGH::NeighborhoodList list(comm,gridData.zoltanPtr.get(),gridData.numPoints,gridData.myGlobalIDs,gridData.myX,horizon);


	// coordinate indices of center cell
	size_t ic = (nx -1)/2;
	size_t jc = ic;
	size_t kc = ic;
	size_t gId = nx * ny * kc + nx * jc + ic;

	/**
	 * WARNING: note following ASSUMPTION -- gId == local id
	 * CAUTION: this test only makes sense in 'serial' -- local id
	 * and gId are not the same in parallel.
	 */
	size_t num_neigh = list.get_num_neigh(gId);

	Array<int> neighborhoodPtr(1+num_neigh);
        Array<double> xPtr(list.get_num_owned_points()*3,list.get_owned_x());
        Array<double> X(3);
        Array<double> Y(3); 
        Array<double> cellVolume(gridData.numPoints,gridData.cellVolume);
        Array<double> yPtr(3*list.get_num_owned_points());
        double gamma = 1.0e-6;
        

        const int *neighborhood = list.get_neighborhood(gId);
	TEST_ASSERT((int)num_neigh == *neighborhood);
	
	/*
	 * expectation is that cell center is at origin
	 */

	TEST_COMPARE(xPtr[3*gId+0], <=,1.0e-15);
	TEST_COMPARE(xPtr[3*gId+1], <=, 1.0e-15);
	TEST_COMPARE(xPtr[3*gId+2], <=, 1.0e-15);

        double m_analytical, m_code, rel_diff, theta;
	
	scf_probe(list, neighborhoodPtr, xPtr, X, Y, cellVolume, yPtr, gamma, m_analytical, m_code, rel_diff, theta, gId, num_neigh );

	
       double tolerance=1.0e-12;
       TEST_COMPARE(theta, <=, tolerance);
       
	/*
	 * PROBE XY
	 */
      probe_shear(MATERIAL_EVALUATION::XY,neighborhoodPtr,X,xPtr,Y,yPtr,cellVolume,horizon,gamma,m_code);
	
	
}

TEUCHOS_UNIT_TEST(NaiveQuadratureConvergenceStudy, n9Test) {

	std::string json_filename = "./input_files/ut_bondVolumeConvergenceStudy_n=9.json";
	set_static_data(json_filename);
        /*
	 * Unit test looks exclusively at the cell at the center of cube;
	 * This cell ID depends upon nx, ny, nz
	 *
	 * MESH INPUT MUST HAVE EVEN NUMBER OF CELLS
	 */
	TEST_ASSERT(0==(nx+1)%2);
	/*
	 * mesh must have equal number of cells along each axis
	 */
	TEST_ASSERT(nx==ny);
	TEST_ASSERT(nx==nz);

	QUICKGRID::QuickGridData gridData = getGrid(json_filename);

	// This load-balances
	gridData = PDNEIGH::getLoadBalancedDiscretization(gridData);

	/*
	 * Create neighborhood with an enlarged horizon
	 */

	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	PDNEIGH::NeighborhoodList list(comm,gridData.zoltanPtr.get(),gridData.numPoints,gridData.myGlobalIDs,gridData.myX,horizon);


	// coordinate indices of center cell
	size_t ic = (nx -1)/2;
	size_t jc = ic;
	size_t kc = ic;
	size_t gId = nx * ny * kc + nx * jc + ic;

	/**
	 * WARNING: note following ASSUMPTION -- gId == local id
	 * CAUTION: this test only makes sense in 'serial' -- local id
	 * and gId are not the same in parallel.
	 */
	size_t num_neigh = list.get_num_neigh(gId);

	Array<int> neighborhoodPtr(1+num_neigh);
        Array<double> xPtr(list.get_num_owned_points()*3,list.get_owned_x());
        Array<double> X(3);
        Array<double> Y(3); 
        Array<double> cellVolume(gridData.numPoints,gridData.cellVolume);
        Array<double> yPtr(3*list.get_num_owned_points());
        double gamma = 1.0e-6;
        

        const int *neighborhood = list.get_neighborhood(gId);
	TEST_ASSERT((int)num_neigh == *neighborhood);
	
	/*
	 * expectation is that cell center is at origin
	 */

	TEST_COMPARE(xPtr[3*gId+0], <=,1.0e-15);
	TEST_COMPARE(xPtr[3*gId+1], <=, 1.0e-15);
	TEST_COMPARE(xPtr[3*gId+2], <=, 1.0e-15);

        double m_analytical, m_code, rel_diff, theta;
	
	scf_probe(list, neighborhoodPtr, xPtr, X, Y, cellVolume, yPtr, gamma, m_analytical, m_code, rel_diff, theta, gId, num_neigh );

	
       double tolerance=1.0e-12;
       TEST_COMPARE(theta, <=, tolerance);
       
	/*
	 * PROBE XY
	 */
      probe_shear(MATERIAL_EVALUATION::XY,neighborhoodPtr,X,xPtr,Y,yPtr,cellVolume,horizon,gamma,m_code);
	
}


TEUCHOS_UNIT_TEST(NaiveQuadratureConvergenceStudy, n11Test) {

	std::string json_filename = "./input_files/ut_bondVolumeConvergenceStudy_n=11.json";
	set_static_data(json_filename);
        /*
	 * Unit test looks exclusively at the cell at the center of cube;
	 * This cell ID depends upon nx, ny, nz
	 *
	 * MESH INPUT MUST HAVE EVEN NUMBER OF CELLS
	 */
	TEST_ASSERT(0==(nx+1)%2);
	/*
	 * mesh must have equal number of cells along each axis
	 */
	TEST_ASSERT(nx==ny);
	TEST_ASSERT(nx==nz);

	QUICKGRID::QuickGridData gridData = getGrid(json_filename);

	// This load-balances
	gridData = PDNEIGH::getLoadBalancedDiscretization(gridData);

	/*
	 * Create neighborhood with an enlarged horizon
	 */

	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	PDNEIGH::NeighborhoodList list(comm,gridData.zoltanPtr.get(),gridData.numPoints,gridData.myGlobalIDs,gridData.myX,horizon);


	// coordinate indices of center cell
	size_t ic = (nx -1)/2;
	size_t jc = ic;
	size_t kc = ic;
	size_t gId = nx * ny * kc + nx * jc + ic;

	/**
	 * WARNING: note following ASSUMPTION -- gId == local id
	 * CAUTION: this test only makes sense in 'serial' -- local id
	 * and gId are not the same in parallel.
	 */
	size_t num_neigh = list.get_num_neigh(gId);

	Array<int> neighborhoodPtr(1+num_neigh);
        Array<double> xPtr(list.get_num_owned_points()*3,list.get_owned_x());
        Array<double> X(3);
        Array<double> Y(3); 
        Array<double> cellVolume(gridData.numPoints,gridData.cellVolume);
        Array<double> yPtr(3*list.get_num_owned_points());
        double gamma = 1.0e-6;
        

        const int *neighborhood = list.get_neighborhood(gId);
	TEST_ASSERT((int)num_neigh == *neighborhood);
	
	/*
	 * expectation is that cell center is at origin
	 */

	TEST_COMPARE(xPtr[3*gId+0], <=,1.0e-15);
	TEST_COMPARE(xPtr[3*gId+1], <=, 1.0e-15);
	TEST_COMPARE(xPtr[3*gId+2], <=, 1.0e-15);

        double m_analytical, m_code, rel_diff, theta;
	
	scf_probe(list, neighborhoodPtr, xPtr, X, Y, cellVolume, yPtr, gamma, m_analytical, m_code, rel_diff, theta, gId, num_neigh );

	
       double tolerance=1.0e-12;
       TEST_COMPARE(theta, <=, tolerance);
       
	/*
	 * PROBE XY
	 */
      probe_shear(MATERIAL_EVALUATION::XY,neighborhoodPtr,X,xPtr,Y,yPtr,cellVolume,horizon,gamma,m_code);
	
}

TEUCHOS_UNIT_TEST(NaiveQuadratureConvergenceStudy, n13Test) {

	std::string json_filename = "./input_files/ut_bondVolumeConvergenceStudy_n=13.json";
	set_static_data(json_filename);
	/*
	 * Unit test looks exclusively at the cell at the center of cube;
	 * This cell ID depends upon nx, ny, nz
	 *
	 * MESH INPUT MUST HAVE EVEN NUMBER OF CELLS
	 */
	TEST_ASSERT(0==(nx+1)%2);
	/*
	 * mesh must have equal number of cells along each axis
	 */
	TEST_ASSERT(nx==ny);
	TEST_ASSERT(nx==nz);

	QUICKGRID::QuickGridData gridData = getGrid(json_filename);

	// This load-balances
	gridData = PDNEIGH::getLoadBalancedDiscretization(gridData);

	/*
	 * Create neighborhood with an enlarged horizon
	 */

	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	PDNEIGH::NeighborhoodList list(comm,gridData.zoltanPtr.get(),gridData.numPoints,gridData.myGlobalIDs,gridData.myX,horizon);


	// coordinate indices of center cell
	size_t ic = (nx -1)/2;
	size_t jc = ic;
	size_t kc = ic;
	size_t gId = nx * ny * kc + nx * jc + ic;

	/**
	 * WARNING: note following ASSUMPTION -- gId == local id
	 * CAUTION: this test only makes sense in 'serial' -- local id
	 * and gId are not the same in parallel.
	 */
	size_t num_neigh = list.get_num_neigh(gId);

	Array<int> neighborhoodPtr(1+num_neigh);
        Array<double> xPtr(list.get_num_owned_points()*3,list.get_owned_x());
        Array<double> X(3);
        Array<double> Y(3); 
        Array<double> cellVolume(gridData.numPoints,gridData.cellVolume);
        Array<double> yPtr(3*list.get_num_owned_points());
        double gamma = 1.0e-6;
        

        const int *neighborhood = list.get_neighborhood(gId);
	TEST_ASSERT((int)num_neigh == *neighborhood);
	
	/*
	 * expectation is that cell center is at origin
	 */

	TEST_COMPARE(xPtr[3*gId+0], <=,1.0e-15);
	TEST_COMPARE(xPtr[3*gId+1], <=, 1.0e-15);
	TEST_COMPARE(xPtr[3*gId+2], <=, 1.0e-15);

        double m_analytical, m_code, rel_diff, theta;
	
	scf_probe(list, neighborhoodPtr, xPtr, X, Y, cellVolume, yPtr, gamma, m_analytical, m_code, rel_diff, theta, gId, num_neigh );

	
       double tolerance=1.0e-12;
       TEST_COMPARE(theta, <=, tolerance);
       
	/*
	 * PROBE XY
	 */
      probe_shear(MATERIAL_EVALUATION::XY,neighborhoodPtr,X,xPtr,Y,yPtr,cellVolume,horizon,gamma,m_code);
}

TEUCHOS_UNIT_TEST(NaiveQuadratureConvergenceStudy, n17Test) {

	std::string json_filename = "./input_files/ut_bondVolumeConvergenceStudy_n=17.json";
	set_static_data(json_filename);

        /*
	 * Unit test looks exclusively at the cell at the center of cube;
	 * This cell ID depends upon nx, ny, nz
	 *
	 * MESH INPUT MUST HAVE EVEN NUMBER OF CELLS
	 */
	TEST_ASSERT(0==(nx+1)%2);
	/*
	 * mesh must have equal number of cells along each axis
	 */
	TEST_ASSERT(nx==ny);
	TEST_ASSERT(nx==nz);

	QUICKGRID::QuickGridData gridData = getGrid(json_filename);

	// This load-balances
	gridData = PDNEIGH::getLoadBalancedDiscretization(gridData);

	/*
	 * Create neighborhood with an enlarged horizon
	 */

	shared_ptr<Epetra_Comm> comm(new Epetra_MpiComm(MPI_COMM_WORLD));
	PDNEIGH::NeighborhoodList list(comm,gridData.zoltanPtr.get(),gridData.numPoints,gridData.myGlobalIDs,gridData.myX,horizon);


	// coordinate indices of center cell
	size_t ic = (nx -1)/2;
	size_t jc = ic;
	size_t kc = ic;
	size_t gId = nx * ny * kc + nx * jc + ic;

	/**
	 * WARNING: note following ASSUMPTION -- gId == local id
	 * CAUTION: this test only makes sense in 'serial' -- local id
	 * and gId are not the same in parallel.
	 */
	size_t num_neigh = list.get_num_neigh(gId);

	Array<int> neighborhoodPtr(1+num_neigh);
        Array<double> xPtr(list.get_num_owned_points()*3,list.get_owned_x());
        Array<double> X(3);
        Array<double> Y(3); 
        Array<double> cellVolume(gridData.numPoints,gridData.cellVolume);
        Array<double> yPtr(3*list.get_num_owned_points());
        double gamma = 1.0e-6;
        

        const int *neighborhood = list.get_neighborhood(gId);
	TEST_ASSERT((int)num_neigh == *neighborhood);
	
	/*
	 * expectation is that cell center is at origin
	 */

	TEST_COMPARE(xPtr[3*gId+0], <=,1.0e-15);
	TEST_COMPARE(xPtr[3*gId+1], <=, 1.0e-15);
	TEST_COMPARE(xPtr[3*gId+2], <=, 1.0e-15);

        double m_analytical, m_code, rel_diff, theta;
	
	scf_probe(list, neighborhoodPtr, xPtr, X, Y, cellVolume, yPtr, gamma, m_analytical, m_code, rel_diff, theta, gId, num_neigh );

	
       double tolerance=1.0e-12;
       TEST_COMPARE(theta, <=, tolerance);
       
	/*
	 * PROBE XY
	 */
      probe_shear(MATERIAL_EVALUATION::XY,neighborhoodPtr,X,xPtr,Y,yPtr,cellVolume,horizon,gamma,m_code);
	
}





/*
 * Dave's Influence Function
 * "x < 0.5 ? 1.0 : -4.0*x*x + 4.0*x"
 */



int main
(
		int argc,
		char* argv[]
)
{


	write_table_1_header("naive_table_1.tex");
//	write_table_2_header("table_2.tex");

	// Initialize UTF
	int flag = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

	close_table_1("naive_table_1.tex");
	return flag;
}

