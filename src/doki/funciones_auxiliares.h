#pragma once
#ifndef MPI_IRLES_H_
#define MPI_IRLES_H_

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h> 
#include <stdarg.h>
#include <mpi.h>

#include "qstate.h"
#include "platform.h"

extern MPI_Datatype MPI_FCOMPLEX;
extern MPI_Datatype MPI_DCOMPLEX;
extern MPI_Datatype MPI_NATURAL_TYPE;

void unpack_int(const int *params, int count, ...);
void unpack_natural_type(const NATURAL_TYPE *params, int count, ...);


unsigned char send_slaves_ints(struct master_state_vector *vector, int count, ...);
unsigned char send_slaves_natural_types(struct master_state_vector *vector, int count, ...);

void recive_confirmations(struct master_state_vector *vector, int *errored);
unsigned char send_ints(int rank, int count, ...);
#endif