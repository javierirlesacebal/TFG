#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h> 
#include <stdarg.h>
#include <mpi.h>


#include "platform.h"
#include "funciones_auxiliares.h"
#include "qstate.h"


void unpack_int(const int *params, int count, ...) {
	int i;
    va_list args;
    va_start(args, count);
    for (i = 0; i < count; i++) {
        int *out_var = va_arg(args, int *);
        *out_var = params[i];
    }
    va_end(args);
}

void unpack_natural_type(const NATURAL_TYPE *params, int count, ...) {
	int i;
    va_list args;
    va_start(args, count);
    for (i = 0; i < count; i++) {
        NATURAL_TYPE *out_var = va_arg(args, NATURAL_TYPE *);
        *out_var = params[i];
    }
    va_end(args);
}

unsigned char send_slaves_ints(struct master_state_vector *vector, int count, ...) {
    int i;
	va_list args;
    int *buffer = malloc(count * sizeof(int));  // Cambiar tipo aquí si usas otro tipo de datos
    if (!buffer){
		return 1;
	}
    va_start(args, count);
    for (i = 0; i < count; i++) {
        buffer[i] = va_arg(args, int);
    }
    va_end(args);
    for (i = 0; i < vector->num_slaves; i++) {
        MPI_Send(buffer, count, MPI_INT, i+1, 0, MPI_COMM_WORLD);
    }
    free(buffer);
	return 0;
}


unsigned char send_slaves_natural_types(struct master_state_vector *vector, int count, ...) {
    int i;
	va_list args;
    NATURAL_TYPE *buffer = malloc(count * sizeof(NATURAL_TYPE));  // Cambiar tipo aquí si usas otro tipo de datos
    if (!buffer){
		return 1;
	}
    va_start(args, count);
    for (i = 0; i < count; i++) {
        buffer[i] = va_arg(args, NATURAL_TYPE);
    }
    va_end(args);
    for (i = 0; i < vector->num_slaves; i++) {
        MPI_Send(buffer, count, MPI_NATURAL_TYPE, i+1, 0, MPI_COMM_WORLD);
    }
    free(buffer);
	return 0;
}

void recive_confirmations(struct master_state_vector *vector, int *errored) {
    int response, i;
    *errored = 0;

    for (i = 0; i < vector->num_slaves; i++) {
        MPI_Recv(&response, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (!response) {
            *errored = 1;
        }
    }
}

unsigned char send_ints(int rank, int count, ...) {
    int i;
	va_list args;
    int *buffer = malloc(count * sizeof(int));
    if (!buffer){
		return 1;
	}
    va_start(args, count);
    for (i = 0; i < count; i++) {
        buffer[i] = va_arg(args, int);
    }
    va_end(args);
    MPI_Send(buffer, count, MPI_INT, rank, 0, MPI_COMM_WORLD);
    free(buffer);
	return 0;
}
