/** \file qstate.h
 *  \brief Functions and structures needed to define a quantum state.
 *  In this file some functions and structures have been defined to create and destroy a quantum state vector.
 */

/** \def __QSTATE_H
 *  \brief Indicates if qstate.h has already been loaded.
 *  If __QSTATE_H is defined, qstate.h file has already been included.
 */

/** \struct array_list qstate.h "qstate.h"
 *  \brief List of complex number arrays.
 *  A list of complex number arrays (chunks).
 */

#pragma once
#ifndef QSTATE_H_
#define QSTATE_H_
#include "platform.h"
#include <stdbool.h>
#include "qgate.h"


struct state_vector {
	/* total size of the vector */
	NATURAL_TYPE size;
	/* number of chunks */
	size_t num_chunks;
	/* number of qubits in this quantum system */
	unsigned int num_qubits;
	/* partial vector */
	COMPLEX_TYPE **vector;
	/* normalization constant */
	REAL_TYPE norm_const;
	/* fcarg initialized */
	bool fcarg_init;
	/* first complex argument */
	REAL_TYPE fcarg;
};

struct master_state_vector {
	// (Numero de elementos del vector 2^num_qubits) y (Numero de qbits del vector de estado)
	NATURAL_TYPE size;
	int num_qubits;
	
	// (Numero de esclavos) y (Punteros a la struct "slave_state_vector" en cada nodo esclavo)
	int num_slaves;
	int *slave_ids;
	
	// (Constante de normalizacion), (primer argumento complejo) y (si fcarg ha sido inicializado)
	REAL_TYPE norm_const;
	REAL_TYPE fcarg;
	bool fcarg_init;
	
};

struct slave_state_vector {
	// (Numero de elementos del vector 2^num_qubits) y (Numero de qbits del vector de estado)
	NATURAL_TYPE size;
	int num_qubits;
	
	// Datos locales almacenados en el vector de estado
	COMPLEX_TYPE *vector;
	
	// (Nº de esclavos asociados al master_state_vector), (rank del esclavo) y (elementos del vector que maneja)
	int num_slaves;
	int assigned_elements;
	
	// (Constante de normalizacion), (primer argumento complejo) y (si fcarg ha sido inicializado)
	REAL_TYPE norm_const;
	REAL_TYPE fcarg;
	bool fcarg_init;
};

/**
	\fn unsigned char state_init(struct state_vector *this, unsigned int num_qubits, int init);
	\brief Initialize a state vector structure.
	\param this Pointer to an already allocated state_vector structure.
	\param num_qubits The number of qubits represented by this state (a maximum of MAX_NUM_QUBITS).
	\param init Whether to initialize to {1, 0, ..., 0} or not.
    \return 0 if ok, 1 if failed to allocate vector, 2 if failed to allocate any chunk, 3 if num_qubits > MAX_NUM_QUBITS.
 */
 
 
// Get element
COMPLEX_TYPE master_state_get(struct master_state_vector *master_state, NATURAL_TYPE i);
void slave_send_vector_element(struct slave_state_vector **local_state_vectors, int state_index);

// State init
unsigned char master_state_init(struct master_state_vector *this, int num_qubits, int init, int available_slaves);
unsigned char slave_state_init(struct slave_state_vector *this, unsigned int num_qubits, bool init, int num_slaves, int slave_rank);
void slave_recive_state_vector(struct slave_state_vector **local_state_vectors);

// State free
void master_state_free(struct master_state_vector *this);

// State clone
unsigned char master_state_clone(struct master_state_vector *destination, struct master_state_vector *source);
unsigned char slave_state_clone(struct slave_state_vector **local_state_vectors);

// Print state
void print_slave_state(struct slave_state_vector **local_state_vectors, int index);


/** \fn unsigned char state_clone(struct state_vector *dest, struct
 * state_vector *source); \brief Clone a state vector structure. \param dest
 * Pointer to an already allocated state_vector structure i which the copy will
 * be stored. \param source Pointer to the state_vector structure that has to
 * be cloned. \return 0 if ok, 1 if failed to allocate dest vector, 2 if failed
 * to allocate any chunk.
 */
 
unsigned char state_init(struct state_vector *this, unsigned int num_qubits, bool init);
unsigned char state_clone(struct state_vector *dest, struct state_vector *source);

void state_clear(struct state_vector *this);

#define state_set(this, i, value) (this)->vector[(i) / COMPLEX_ARRAY_SIZE][(i) % COMPLEX_ARRAY_SIZE] = value

#define state_get(this, i) (COMPLEX_DIV_R((this)->vector[(i) / COMPLEX_ARRAY_SIZE][(i) % COMPLEX_ARRAY_SIZE], (this)->norm_const))
#define slave_state_get_normalized(state_index, reg_index) COMPLEX_DIV_R(local_state_vectors[(state_index)]->vector[(reg_index)], local_state_vectors[(state_index)]->norm_const)

size_t state_mem_size(struct state_vector *this);
#endif /* QSTATE_H_ */
