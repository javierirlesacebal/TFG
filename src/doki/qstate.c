#include "qstate.h"
#include "platform.h"
#include "funciones_auxiliares.h"
#include <stdbool.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

extern MPI_Datatype MPI_DCOMPLEX;
extern MPI_Datatype MPI_NATURAL_TYPE;

// Revisada
COMPLEX_TYPE master_state_get(struct master_state_vector *master_state, NATURAL_TYPE i){
	NATURAL_TYPE elements_per_slave;
	int target_slave;
	COMPLEX_TYPE element;
	
	elements_per_slave = master_state->size / master_state->num_slaves;
	target_slave = i / elements_per_slave;
	i = i & (elements_per_slave - 1);
	
	send_ints(target_slave + 1, 2, 12, master_state->slave_ids[target_slave]);
	MPI_Send(&i, 1, MPI_NATURAL_TYPE, target_slave + 1, 0, MPI_COMM_WORLD);
	MPI_Recv(&element, 1, MPI_DCOMPLEX, target_slave + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	return element;
}

// Revisada
void slave_send_vector_element(struct slave_state_vector **local_state_vectors, int state_index){
	NATURAL_TYPE i;

	MPI_Recv(&i, 1, MPI_NATURAL_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Send(&local_state_vectors[state_index]->vector[i], 1, MPI_DCOMPLEX, 0, 0, MPI_COMM_WORLD);
}

// Revisada
unsigned char master_state_init(struct master_state_vector *this, int num_qubits, int init, int available_slaves) {  // REVISION FINAL
	unsigned char errored = 0;
	int response[2];
	int i;

	if (this == NULL){
		return 5;
	}
	// Comprobar que no se sobrepasa el numero maximo de Qubits
	if (num_qubits > MAX_NUM_QUBITS) {
		errored = 1;
		goto cleanup;
	}
	
	// Solo acepta un numero de esclavos potencia exacta de 2
	if (!(available_slaves) || ( available_slaves & (available_slaves-1))){
		errored = 2;
		goto cleanup;
	}
	
	// Se inicializan las variables simples
	this->size = NATURAL_ONE << num_qubits;
	this->num_qubits = num_qubits;
	this->norm_const = 1;
	this->fcarg = -10.0;
	this->fcarg_init = 0;
	
	// Calcula el numero de esclavos necesarios
	if (num_qubits > MIN_SV_ELEMENTS){
		this->num_slaves = MIN((1 << (num_qubits - MIN_SV_ELEMENTS)), available_slaves);
	} else {
		this->num_slaves = 1;
	}
	
	// Inicializa el vector de punteros de esclavos
	SAFE_MALLOC(this->slave_ids, this->num_slaves, int, errored, 1, cleanup);
	
	// Marca si ha habido algun error
	send_slaves_ints(this, 2, 1, 0);
	send_slaves_ints(this, 3, this->num_qubits, init, this->num_slaves);
	// Manda a crear los "slave_state_vector"
	for (i = 0; i < this->num_slaves; i++) {
		MPI_Recv(&response, 2, MPI_INT, i+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (response[0]){
			this->slave_ids[i] = response[1];
		} else {
			errored = 3;
		}
	}
cleanup:
	if (errored){
		master_state_free(this);
		return errored;
	}
	return 0;
}

// Revisado
unsigned char slave_state_init(struct slave_state_vector *this, unsigned int num_qubits, bool init, int num_slaves, int slave_rank) {  // REVISION FINAL
	unsigned char errored = 0;
	int slave_id;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &slave_id);
	slave_id = slave_id-1;
	
	// Se inicializan las variables simples
	this->size = NATURAL_ONE << num_qubits;
	this->num_qubits = num_qubits;
	this->num_slaves = num_slaves;
	this->norm_const = 1;
	this->fcarg = -10.0;
	this->fcarg_init = 0;
	this->assigned_elements = this->size / num_slaves;
	this->vector = NULL;
	
	// Se reserva memoria para los datos del vector
	if (init) {
		SAFE_CALLOC(this->vector, this->assigned_elements, COMPLEX_TYPE, errored, 1, cleanup);
	} else {
		SAFE_MALLOC(this->vector, this->assigned_elements, COMPLEX_TYPE, errored, 1, cleanup);
	}
	
	// Si debe inicializarse y contiene la primera parte del vector
	if (init == 1 && slave_id == 0) {
		this->vector[0] = COMPLEX_ONE;
	}
	
cleanup:
	if (errored){
		printf("Slave_id %d FAILED FAILED FAILED TO ALLOCATE AMPLITUDE VECTOR\n", slave_id);
		SAFE_FREE(this->vector);
		return errored;
	}
	return 0;
}

// Revisada
void master_state_free(struct master_state_vector *this) {
	int i;
	
	if (this == NULL){
		return;
	}
	if (this->slave_ids != NULL){
		for (i=0; i < this->num_slaves; i++){
			if (this->slave_ids[i] != -1){
				send_ints(i+1, 2, 2, this->slave_ids[i]);
			}
		}
		SAFE_FREE(this->slave_ids);
	}
	SAFE_FREE(this);
}


// Revisada
void slave_recive_state_vector(struct slave_state_vector **local_state_vectors){
	NATURAL_TYPE i, offset;
	int parameters[3];
	int state_index, num_chunks, chunk_size;
	
	MPI_Recv(parameters, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	unpack_int(parameters, 3, &state_index, &num_chunks, &chunk_size);
	
	offset = 0;
	for (i = 0; i < num_chunks; i++) {
		MPI_Recv(&local_state_vectors[state_index]->vector[offset], chunk_size, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		offset+=chunk_size;
	}
}

// Revisada
unsigned char master_state_clone(struct master_state_vector *destination, struct master_state_vector *source) {
	unsigned char exit_code;
	int i;
	
	exit_code = master_state_init(destination, source->num_qubits, false, source->num_slaves);
	
	if (exit_code != 0) {
		return exit_code;
	}

	// Manda a los esclavos a entrar en "slave_state_clone()"
	send_slaves_ints(destination, 2, 6, 0);
	
	// Manda a los esclavos que vector clonar sobre cual
	for (i = 0; i < destination->num_slaves; i++) {
		send_ints(i+1, 2, source->slave_ids[i], destination->slave_ids[i]); // #SEW
	}
	
	return 0;
}


// Revisado
unsigned char slave_state_clone(struct slave_state_vector **local_state_vectors) { // REVISION FINAL
	int destination, source;
	int parameters[2];
	
	// Obtener los parametros
	MPI_Recv(&parameters, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // #SEW
	unpack_int(parameters, 2, &source, &destination);
	
	// Copiar el vector de estado
	memcpy(local_state_vectors[destination]->vector, local_state_vectors[source]->vector, local_state_vectors[source]->assigned_elements * sizeof(COMPLEX_TYPE));

	// Establecer el resto de parametros
	local_state_vectors[destination]->norm_const = local_state_vectors[source]->norm_const;
	local_state_vectors[destination]->fcarg = local_state_vectors[source]->fcarg;
	local_state_vectors[destination]->fcarg_init = local_state_vectors[source]->fcarg_init;
	return 0;
}

// Revisado
void print_slave_state(struct slave_state_vector **local_state_vectors, int index){
	NATURAL_TYPE i, element_index;
	COMPLEX_TYPE element;
	int slave_id;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &slave_id);
	slave_id = slave_id - 1;
	
	printf("SLAVE_id %d: elementos asignados(%d)\n", slave_id, (int) local_state_vectors[index]->assigned_elements);
	for(i = 0; i < local_state_vectors[index]->assigned_elements; i++){
		element_index = local_state_vectors[index]->assigned_elements * slave_id + i;
		element = local_state_vectors[index]->vector[i];
		printf("SLAVE_id %d: printing state_vector[%d]global[%zu](%.5f+%.5fi)\n", slave_id, index, element_index, RE(element), IM(element));
	}
}


unsigned char state_init(struct state_vector *this, unsigned int num_qubits, bool init) {
	size_t i, offset, errored_chunk;
	bool errored;

	if (num_qubits > MAX_NUM_QUBITS) {
		return 3;
	}
	this->size = NATURAL_ONE << num_qubits;
	this->fcarg_init = 0;
	this->fcarg = -10.0;
	this->num_qubits = num_qubits;
	this->norm_const = 1;
	this->num_chunks = this->size / COMPLEX_ARRAY_SIZE;
	offset = this->size % COMPLEX_ARRAY_SIZE;
	
	if (offset > 0) {
		this->num_chunks++;
	} else {
		offset = COMPLEX_ARRAY_SIZE;
	}
	this->vector = MALLOC_TYPE(this->num_chunks, COMPLEX_TYPE *);
	if (this->vector == NULL) {
		return 1;
	}
	errored = 0;
	for (i = 0; i < this->num_chunks - 1; i++) {
		if (init) {
			this->vector[i] = CALLOC_TYPE(COMPLEX_ARRAY_SIZE, COMPLEX_TYPE);
		} else {
			this->vector[i] = MALLOC_TYPE(COMPLEX_ARRAY_SIZE, COMPLEX_TYPE);
		}
		if (this->vector[i] == NULL) {
			errored_chunk = i;
			errored = 1;
			break;
		}
	}
	if (!errored) {
		if (init) {
			this->vector[this->num_chunks - 1] = CALLOC_TYPE(offset, COMPLEX_TYPE);
		} else {
			this->vector[this->num_chunks - 1] = MALLOC_TYPE(offset, COMPLEX_TYPE);
		}
		if (this->vector[this->num_chunks - 1] == NULL) {
			errored = 1;
			errored_chunk = this->num_chunks - 1;
		}
	}
	if (errored) {
		for (i = 0; i < errored_chunk; i++) {
			free(this->vector[i]);
		}
		free(this->vector);
		return 2;
	}
	if (init) {
		this->vector[0][0] = COMPLEX_ONE;
	}

	return 0;
}

unsigned char state_clone(struct state_vector *dest, struct state_vector *source) {
	NATURAL_TYPE i;
	unsigned char exit_code;
	exit_code = state_init(dest, source->num_qubits, false);
	if (exit_code != 0) {
		return exit_code;
	}
#pragma omp parallel for default(none) \
	shared(source, dest, COMPLEX_ARRAY_SIZE) private(i)
	for (i = 0; i < source->size; i++) {
		state_set(dest, i, state_get(source, i));
	}
	return 0;
}

void state_clear(struct state_vector *this) {
	size_t i;
	if (this->vector != NULL) {
		for (i = 0; i < this->num_chunks; i++) {
			free(this->vector[i]);
		}
		free(this->vector);
	}
	this->vector = NULL;
	this->num_chunks = 0;
	this->num_qubits = 0;
	this->size = 0;
	this->norm_const = 0.0;
}

size_t state_mem_size(struct state_vector *this) {
	size_t state_size;
	if (this == NULL) {
		return 0;
	}
	state_size = sizeof(struct state_vector);
	state_size += this->num_chunks * sizeof(COMPLEX_TYPE *);
	state_size += (this->num_chunks - 1) * COMPLEX_ARRAY_SIZE *
		      sizeof(COMPLEX_TYPE);
	state_size += (this->size % COMPLEX_ARRAY_SIZE) * sizeof(COMPLEX_TYPE);
	return state_size;
}
