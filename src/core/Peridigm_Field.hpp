/*! \file Peridigm_Field.hpp */

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

#ifndef PERIDIGM_FIELD_HPP
#define PERIDIGM_FIELD_HPP

#include <Teuchos_Assert.hpp>
#include <Teuchos_Exceptions.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <map>

namespace PeridigmNS{

namespace PeridigmField {

  enum Relation {
    UNDEFINED_RELATION=0,
    NODE,
    ELEMENT,
    BOND,
    GLOBAL
  };

  // Length corresponds to element size in an Epetra_Multivector.
  // Note that entries are cast to int elsewhere in the code, so
  // entries must have logically consistent values.
  enum Length {
    UNDEFINED_LENGTH = 0,
    LENGTH_1,
    LENGTH_2,
    LENGTH_3,
    LENGTH_4,
    LENGTH_5,
    LENGTH_6,
    LENGTH_7,
    LENGTH_8,
    LENGTH_9,
    SCALAR,
    VECTOR,
    SYMMETRIC_TENSOR,
    FULL_TENSOR
  };

  enum Temporal {
    UNDEFINED_TEMPORAL=0,
    CONSTANT,
    TWO_STEP
  };

  enum Step {
    UNDEFINED_STEP=0,
    STEP_NONE,
    STEP_N,
    STEP_NP1
  };

  //! Return the integer value of the field length for a given PeridigmField::Length.
  int variableDimension(Length length);

} // namespace PeridigmField

class FieldSpec {

public:

  //! Constructor.
  explicit FieldSpec()
    : relation(PeridigmField::UNDEFINED_RELATION),
      length(PeridigmField::UNDEFINED_LENGTH),
      temporal(PeridigmField::UNDEFINED_TEMPORAL),
      label("Undefined"),
      id(-1) {}

  //! Default constructor
  explicit FieldSpec(PeridigmField::Relation relation_,
                     PeridigmField::Length length_,
                     PeridigmField::Temporal temporal_,
                     std::string label_,
                     int id_)
    : relation(relation_), length(length_), temporal(temporal_), label(label_), id(id_) {}

  bool operator == (const FieldSpec& right) const { return (id == right.id); }
  bool operator != (const FieldSpec& right) const { return (id != right.id); }
  bool operator < (const FieldSpec& right) const { return (id < right.id); }

  PeridigmField::Relation getRelation() const { return relation; }
  PeridigmField::Length getLength() const { return length; }
  PeridigmField::Temporal getTemporal() const { return temporal; }
  std::string getLabel() const { return label; }
  int getId() const { return id; }

  PeridigmField::Relation relation;
  PeridigmField::Length length;
  PeridigmField::Temporal temporal;
  std::string label;
  int id;

  //! Destructor.
  ~FieldSpec(){}
};

class FieldManager {

public:

  //! Singleton.
  static FieldManager & self();

  int getFieldId(PeridigmField::Relation relation_,
                 PeridigmField::Length length_,
                 PeridigmField::Temporal temporal_,
                 std::string label_) ;

  bool hasField(std::string label);

  int getFieldId(std::string label);

  FieldSpec getFieldSpec(int fieldId);

  FieldSpec getFieldSpec(std::string label);

  std::vector<FieldSpec> getFieldSpecs() { return fieldSpecs; }

  bool isGlobalSpec(int fieldId) { return specIsGlobal[fieldId]; }

  std::vector<std::string> getFieldLabels() {
    std::vector<std::string> labels;
    for(std::vector<FieldSpec>::const_iterator it = fieldSpecs.begin() ; it != fieldSpecs.end() ; it++)
      labels.push_back(it->label);
    return labels;
  }

  std::vector<FieldSpec> getGlobalFieldSpecs() {
    std::vector<FieldSpec> specs;
    for(std::vector<FieldSpec>::const_iterator it = fieldSpecs.begin() ; it != fieldSpecs.end() ; it++){
      if(it->getRelation() == PeridigmField::GLOBAL)
        specs.push_back(*it);
    }
    return specs;
  }

  void printFieldSpecs(std::ostream& os) {
    std::stringstream ss;
    ss << "Field specifications:";
    for(std::vector<FieldSpec>::const_iterator it = fieldSpecs.begin() ; it != fieldSpecs.end() ; it++){
      ss << "\n";
      ss << "\n  Label:    " << it->label;
      ss << "\n  Relation: " << it->relation;
      ss << "\n  Length:   " << it->length;
      ss << "\n  Temporal  " << it->temporal;
      ss << "\n  Id:       " << it->id;
    }
    os << ss.str() << std::endl;
  }

  void printFieldLabels(std::ostream& os) {
    std::vector<std::string> labels = getFieldLabels();
    std::string msg("\nField labels:");
    for(unsigned int i=0 ; i<labels.size() ; ++i)
      msg += "\n  " + labels[i];
    os << msg << "\n" << std::endl;
  }

private:

  //! Constructor, private to prevent use (singleton class).
  FieldManager(){}

  //! Private and unimplemented to prevent use
  FieldManager( const FieldManager & );

  //! Private and unimplemented to prevent use
  FieldManager & operator= ( const FieldManager & );

  std::vector<FieldSpec> fieldSpecs;

  std::map<std::string, int> labelToIdMap;

  std::vector<bool> specIsGlobal;
};

}

std::ostream& operator<<(std::ostream& os, const PeridigmNS::PeridigmField::Relation& step);

std::ostream& operator<<(std::ostream& os, const PeridigmNS::PeridigmField::Length& step);

std::ostream& operator<<(std::ostream& os, const PeridigmNS::PeridigmField::Temporal& step);

std::ostream& operator<<(std::ostream& os, const PeridigmNS::PeridigmField::Step& step);

std::ostream& operator<<(std::ostream& os, const PeridigmNS::FieldSpec& fieldSpec);

#endif // PERIDIGM_FIELD_HPP
