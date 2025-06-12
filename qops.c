#include <Python.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <inttypes.h> 
#include <stdarg.h>
#include "platform.h"
#include "qgate.h"
#include "qops.h"
#include "qstate.h"
#include "mpi_irles.h"


#define SET_BIT(number, index, value) (((number) & ~(1U << (index))) | (((value) & 1U) << (index)))
#define COMPARE_BIT_REMOVAL(v1, v2, x) ((((v1) & ((1ULL << (x)) - 1)) | (((v1) >> 1) & ~((1ULL << (x)) - 1))) == ((v2) >> 1))
#define REMOVE_BIT_AND_SHIFT_HIGH(v1, x) (((v1) & ((1ULL << (x)) - 1)) | (((v1) >> 1) & ~((1ULL << (x)) - 1)))
#define INSERT_BIT_AT_INDEX_SHIFTLEFT(number, index, value) ((((number) & ~((1ULL << (index)) - 1)) << 1) | (((value) & 1ULL) << (index)) | ((number) & ((1ULL << (index)) - 1)))
#define INSERT_BIT_AT_INDEX_SHIFTRIGHT(number, index, value) (((number) & ~((1ULL << ((index) + 1)) - 1)) | (((value) & 1ULL) << (index)) | (((number) & ((1ULL << (index+1)) - 1)) >> 1))


extern MPI_Datatype MPI_DCOMPLEX;
extern MPI_Datatype MPI_NATURAL_TYPE;

static size_t size_state_capsule(void *raw_capsule);



// Revisado
REAL_TYPE get_master_global_phase(struct master_state_vector *master_state) {
	NATURAL_TYPE i;
	REAL_TYPE phase;
	COMPLEX_TYPE val;
	if (master_state->fcarg_init) {
		return master_state->fcarg;
	}
	phase = 0.0;
	for (i = 0; i < master_state->size; i++) {
		val = master_state_get(master_state, i);
		if (RE(val) == 0. && IM(val) == 0.) {
			continue;
		}
		if (IM(val) != 0.) {
			phase = ARG(val);
		}
		break;
	}
	master_state->fcarg = phase;
	master_state->fcarg_init = 1;
	return phase;
}

// Revisado
unsigned char slave_state_normalize(struct slave_state_vector **local_state_vectors, int state_index){
	REAL_TYPE norm_const, inv_norm_const;
	REAL_TYPE sq_sum;
	NATURAL_TYPE i;
	
	// Calcula la constante de normalizacion parcial
	sq_sum = 0;
	for (i = 0; i < local_state_vectors[state_index]->assigned_elements; i++){
		sq_sum += pow(RE(local_state_vectors[state_index]->vector[i]), 2) + pow(IM(local_state_vectors[state_index]->vector[i]), 2);
	}
	
	// La comparte y obtiene la constante de normalizacion total
	MPI_Send(&sq_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	MPI_Recv(&norm_const, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	inv_norm_const = 1 / norm_const;
	for (i = 0; i < local_state_vectors[state_index]->assigned_elements; i++){
		local_state_vectors[state_index]->vector[i] = COMPLEX_MULT_R(local_state_vectors[state_index]->vector[i], inv_norm_const);
	}
	local_state_vectors[state_index]->norm_const = NATURAL_ONE;
	return 0;
}

// Revisado
unsigned char normalize_master_state(struct master_state_vector *master_state){
	REAL_TYPE norm_const, slave_norm_const;
	int parameters[2];
	int i;

	// Manda a los esclavos a entrar en "slave_state_normalize()" y el 
	parameters[0] = 11;
	for (i = 0; i<master_state->num_slaves; i++){
		parameters[1] = (int) master_state->slave_ids[i];
		MPI_Send(parameters, 2, MPI_INT, i+1, 0, MPI_COMM_WORLD);
	}
	
	// Recoge la 'norm_const' de cada esclavo
	norm_const = 0;
	for (i = 0; i<master_state->num_slaves; i++){
		MPI_Recv(&slave_norm_const, 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		norm_const += slave_norm_const;
	}

	// Envia a cada esclavo la 'norm_const' final
	master_state->norm_const = sqrt(norm_const);
	for (i = 0; i<master_state->num_slaves; i++){
		MPI_Send(&master_state->norm_const, 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD);
	}

	return 0;
}

// Revisado
unsigned char apply_gate_to_master_state(struct master_state_vector *master_state, struct master_state_vector *new_master_state, int num_targets, unsigned int *targets, unsigned int num_controls,unsigned int *controls, unsigned int num_anticontrols, unsigned int *anticontrols) {
	NATURAL_TYPE control_mask, anticontrol_mask, msb_targets_mask, target_mask_inv = NATURAL_ZERO;
	NATURAL_TYPE i;
	int j, response, num_lsb_targets, msb_bitsize, lsb_bitsize, state_slave_id, new_state_slave_id;
	int num_msb_targets = 0, errored = 0; 
	unsigned char exit_code;
	size_t init = 2;
	
	// Inicializa el new_master_state
	exit_code = master_state_init(new_master_state, master_state->num_qubits, init, master_state->num_slaves);
	if (exit_code){
		errored = 1;
		goto cleanup;
	}

	// Calcula las variables necesarias para la comunicacion entre esclavos
	msb_bitsize = ffs(master_state->num_slaves) - 1;
	lsb_bitsize = master_state->num_qubits - msb_bitsize;
	for (i = 0; i < num_targets; i++){
		num_msb_targets += (targets[i] >= lsb_bitsize);
	}
	num_lsb_targets = num_targets - num_msb_targets;	
	
	// Generar las mascaras
	control_mask = NATURAL_ZERO;
	for (j = 0; j < num_controls; j++){
		control_mask |= NATURAL_ONE << controls[j];
	}
	anticontrol_mask = NATURAL_ZERO;
	for (j = 0; j < num_anticontrols; j++){
		anticontrol_mask |= NATURAL_ONE << anticontrols[j];
	}
	msb_targets_mask = NATURAL_ZERO;
	for (i = 0; i < num_targets; i++){
		if(targets[i] >= lsb_bitsize){
			msb_targets_mask = SET_BIT(msb_targets_mask, targets[i]-lsb_bitsize, 1);
		}
		target_mask_inv |= (NATURAL_ONE << targets[i]);
	}
	target_mask_inv = ~target_mask_inv;
	
	// Envia para que los esclavos entren en "apply_gate_to_slave_state()"
	send_slaves_ints(new_master_state, 2, 4, 0);
	
	// Envia a todos los esclavos las ids de los vectores sobre los que trabajar y el numero de targets y las mascaras
	for (i = 0; i<new_master_state->num_slaves; i++){
		state_slave_id = (int) master_state->slave_ids[i];
		new_state_slave_id = (int) new_master_state->slave_ids[i];
		send_ints(i+1, 7, state_slave_id, new_state_slave_id, num_targets, num_msb_targets, num_lsb_targets, msb_bitsize, lsb_bitsize);
	}
	send_slaves_natural_types(new_master_state, 4, control_mask, anticontrol_mask, msb_targets_mask, target_mask_inv);

	// RECIBIR CONFIRMACION DE QUE HA IDO BIEN DESPUES DE RESERVAR LA MEMORIA DE targets y tal
	recive_confirmations(new_master_state, &errored);
	response = !errored;
	send_slaves_ints(new_master_state, 1, &response);

	// Si algun esclavo ha fallado, indica al resto que aborten la apply_gate_to_slave_state
	if (errored){
		errored = 2;
		goto cleanup;
	}
	
	// Envia los targets
	MPI_SEND_TO_SLAVES(new_master_state, targets, num_targets, MPI_UNSIGNED);
	
	// Normaliza en caso de error por decimales
	exit_code = normalize_master_state(new_master_state);
	if (exit_code){
		errored = 3;
	}
	
cleanup:
	if (errored){
		master_state_free(new_master_state);
		return (unsigned char) errored;
	}
	return 0;
}

// Revisada
unsigned char apply_gate_to_slave_state(struct slave_state_vector **local_state_vectors, struct qgate *gate) {
	NATURAL_TYPE block_size, chunk_index, chunk_offset, chunk, chunk_size;
	NATURAL_TYPE control_mask, anticontrol_mask, target_mask_inv, msb_targets_mask;
	NATURAL_TYPE global_offset, global_index, reg_index, lut_index, fixed_index, local_index;
	COMPLEX_TYPE sum, gate_value, vector_value;
	int state_index, new_state_index;
	int num_targets, num_msb_targets, num_lsb_targets;
	int msb_bitsize, lsb_bitsize;
	int gate_lsb_size, gate_offset;
	int slave_id, communication_id, errored, response, target_index, row, value;
	int target, round, local_gate_permutation;
	NATURAL_TYPE masks[4];
	int parameters1[7];
	COMPLEX_TYPE *chunk_to_send = NULL, *chunk_to_recive = NULL;
	NATURAL_TYPE *lut = NULL;
	unsigned int *targets = NULL;

	// Parametros de MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &slave_id);
	slave_id = slave_id-1;
	errored = 0;
	
	// Recibe los parametros necesiarios
	MPI_Recv(parameters1, 7, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	unpack_int(parameters1, 7, &state_index, &new_state_index, &num_targets, &num_msb_targets, &num_lsb_targets, &msb_bitsize, &lsb_bitsize); 
	MPI_Recv(masks, 4, MPI_NATURAL_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	unpack_natural_type(masks, 4, &control_mask, &anticontrol_mask, &msb_targets_mask, &target_mask_inv);
	
	// Si el vector de estado sobre el que se aplica la puerta es NULL, devuelve error
	if (local_state_vectors[state_index] == NULL || local_state_vectors[new_state_index] == NULL){
		errored = 1;
		goto cleanup;
	}
	
	// Calcula las variables basicas
	block_size = (NATURAL_ONE << lsb_bitsize);
	chunk_size = MIN(block_size, SLAVE_COMMUNICATION_CHUNK_SIZE);
	gate_lsb_size = 1 << num_lsb_targets;
	
	// Reserva memoria para los targets
	SAFE_MALLOC(targets, num_targets, unsigned int, errored, 1, cleanup);
	SAFE_MALLOC(lut, gate_lsb_size, NATURAL_TYPE, errored, 1, cleanup);
	SAFE_MALLOC_CONDITIONAL(chunk_to_send, chunk_size, COMPLEX_TYPE, errored, 1, cleanup, num_msb_targets);
	SAFE_MALLOC_CONDITIONAL(chunk_to_recive, chunk_size, COMPLEX_TYPE, errored, 1, cleanup, num_msb_targets);
	
	// Envia confirmacion positiva, y espera la confirmacion del maestro
	response = 1;
	MPI_Send(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); // #AAA
	MPI_Recv(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);// #BBB
	if (!response) {
		goto cleanup;
	}
	
	// Recibe los targets
	MPI_Recv(targets, num_targets, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// Calcula la lut
	for (lut_index = 0; lut_index < gate_lsb_size; ++lut_index) {
		reg_index = 0, target_index = 0;
		for (target = 0; target < num_targets; target++) {
			if (targets[target] < lsb_bitsize){
				reg_index |= (((lut_index >> target_index) & NATURAL_ONE) << targets[target_index]);
				target_index++;
			}
		}
		lut[lut_index] = reg_index;
	}
	
	// Calcula la gate_offset
	gate_offset = 0, target_index = 0;		
	for (target = 0; target < num_targets; target++){
		if (targets[target] >= lsb_bitsize){
			value = slave_id >> (targets[target_index] - lsb_bitsize) & 1;
			gate_offset = SET_BIT(gate_offset, target_index, value);
			target_index++;
		}
	}

	// EMPIEZA ZONA PRIVADA
	global_offset = block_size * slave_id;
	for (local_index = 0; local_index < block_size; local_index++) {
		global_index = global_offset + local_index;
		if (((global_index & control_mask) != control_mask || (global_index & anticontrol_mask) != 0)) {
			local_state_vectors[new_state_index]->vector[local_index] = local_state_vectors[state_index]->vector[local_index];
		} else {
			row = 0;
			for (target = 0; target < num_targets; target++) {
				row += ((global_index & (NATURAL_ONE << targets[target])) != 0) << target;
			}
			fixed_index = local_index & target_mask_inv;
			sum = COMPLEX_ZERO;
			for (local_gate_permutation = 0; local_gate_permutation < gate_lsb_size; local_gate_permutation++) {
				reg_index = fixed_index | lut[local_gate_permutation];
				gate_value = gate->matrix[row][local_gate_permutation+gate_offset];
				vector_value = local_state_vectors[state_index]->vector[reg_index];
				sum = COMPLEX_ADD(sum, COMPLEX_MULT(vector_value, gate_value));
			}
			local_state_vectors[new_state_index]->vector[local_index] = sum;
		}
	}

	
	
	// EMPIEZA ZONA COMUN
	for (round = 0; round < local_state_vectors[state_index]->num_slaves; round++){
		communication_id = (round - slave_id + local_state_vectors[state_index]->num_slaves) % local_state_vectors[state_index]->num_slaves;
		if ((communication_id == slave_id) || ((communication_id^slave_id) & ~msb_targets_mask)){
			continue;
		}	
		global_offset = block_size * communication_id, chunk_offset = 0, chunk_index = 0;
		for (local_index = 0; local_index < block_size; local_index++) {
			global_index = global_offset + local_index;
			fixed_index = local_index & target_mask_inv;
			if ((global_index & control_mask) != control_mask || (global_index & anticontrol_mask) != 0) {
				chunk_to_send[chunk_index] = COMPLEX_ZERO;
			} else {
				row = 0, sum = COMPLEX_ZERO;
				for (target = 0; target < num_targets; target++) {
					row += ((global_index & (NATURAL_ONE << targets[target])) != 0) << target;
				}
				for (local_gate_permutation = 0; local_gate_permutation < gate_lsb_size; local_gate_permutation++) {
					reg_index = fixed_index | lut[local_gate_permutation];	
					gate_value = gate->matrix[row][gate_offset+local_gate_permutation];
					vector_value = local_state_vectors[state_index]->vector[reg_index];
					sum = COMPLEX_ADD(sum, COMPLEX_MULT(vector_value, gate_value));
				}
				chunk_to_send[chunk_index] = sum;	
			}
			chunk_index++;
			if (chunk_index == chunk_size){
				if (communication_id > slave_id){
					MPI_Send(chunk_to_send,   chunk_size, MPI_DOUBLE_COMPLEX, communication_id+1, 0, MPI_COMM_WORLD);
					MPI_Recv(chunk_to_recive, chunk_size, MPI_DOUBLE_COMPLEX, communication_id+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				} else {
					MPI_Recv(chunk_to_recive, chunk_size, MPI_DOUBLE_COMPLEX, communication_id+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Send(chunk_to_send,   chunk_size, MPI_DOUBLE_COMPLEX, communication_id+1, 0, MPI_COMM_WORLD);
				}
				for (chunk = 0; chunk < chunk_size; chunk++){
					local_state_vectors[new_state_index]->vector[chunk_offset+chunk] = COMPLEX_ADD(chunk_to_recive[chunk], local_state_vectors[new_state_index]->vector[chunk_offset+chunk]);
				}
				chunk_index = 0;
				chunk_offset += chunk_size;
			}
		}
	}
	
cleanup:
	SAFE_FREE(targets);
	SAFE_FREE(lut);
	SAFE_FREE(chunk_to_send);
	SAFE_FREE(chunk_to_recive);
	
	if (errored){
		response = 0;
		MPI_Send(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	return 0;
}

// Revisada
void join_process_self_main(struct slave_state_vector *result_vector, struct slave_state_vector *main_vector, NATURAL_TYPE main_slave_size){
	NATURAL_TYPE i, j, offset, times;

	times = result_vector->assigned_elements / main_slave_size;
	for(i = 0; i < main_slave_size; i++){
		offset = i*times;
		for(j = 0; j < times; j++){
			result_vector->vector[offset+j] = main_vector->vector[i];
		}
	}
}

// Revisada
void join_send_main(struct slave_state_vector *main_vector, int send_index, int mtrsr, NATURAL_TYPE num_chunks, NATURAL_TYPE chunk_size){
	NATURAL_TYPE i, offset, g_offset;

	g_offset = ((send_index % mtrsr) * main_vector->assigned_elements) / mtrsr;
	for(i = 0; i < num_chunks; i++){
		MPI_Send(&main_vector->vector[g_offset + i * chunk_size], chunk_size, MPI_DOUBLE_COMPLEX, send_index+1, 0, MPI_COMM_WORLD); //#KVP
	}
}


// Revisada
void join_recive_main(struct slave_state_vector *result_vector, COMPLEX_TYPE *recived_main_chunk, NATURAL_TYPE main_chunk_size, NATURAL_TYPE main_slave_size, int recive_from, NATURAL_TYPE num_main_chunks){
	NATURAL_TYPE i, j, k, offset, times;
	times = result_vector->assigned_elements / main_slave_size;
	for(k = 0; k < num_main_chunks; k++){
		MPI_Recv(recived_main_chunk, main_chunk_size, MPI_DOUBLE_COMPLEX, recive_from+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //#KVP
		for(i = 0; i < main_chunk_size; i++){
			offset = (k * main_chunk_size + i) * times;
			for(j = 0; j < times; j++){
				result_vector->vector[offset+j] = recived_main_chunk[i];
			}
		}
	}
}

// Revisada
void join_process_self_secondary(struct slave_state_vector *result, struct slave_state_vector *secondary_vector){
	NATURAL_TYPE i, j, start;
	COMPLEX_TYPE aux;
	int slave_id;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &slave_id);
	slave_id = slave_id - 1;

	start = (slave_id * secondary_vector->assigned_elements) - ((slave_id << log2_64((uint64_t) result->assigned_elements)) % (secondary_vector->assigned_elements * secondary_vector->num_slaves));
	if(start < 0){
		return;
	}
	
	for(i = start; i < result->assigned_elements; i=i+secondary_vector->size){
		for(j = 0; j < secondary_vector->assigned_elements; j++){
			result->vector[i+j] = COMPLEX_MULT(result->vector[i+j], secondary_vector->vector[j]);
		}
	}
}

// Revsiado
void join_send_secondary(struct slave_state_vector *secondary, int send_index, NATURAL_TYPE num_secondary_chunks, NATURAL_TYPE secondary_chunk_size, NATURAL_TYPE secondary_num_slaves){
	NATURAL_TYPE k;
	int mpi_rank;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	if (mpi_rank > secondary_num_slaves){
		return;
	}
	
	for(k = 0; k < num_secondary_chunks; k++){
		MPI_Ssend(&secondary->vector[k*secondary_chunk_size], secondary_chunk_size, MPI_DOUBLE_COMPLEX, send_index+1, 0, MPI_COMM_WORLD); // #POK
	}
}


// Revisado
void join_recive_secondary(struct slave_state_vector *result_vector, COMPLEX_TYPE * recived_secondary_chunk, int recive_index, NATURAL_TYPE num_secondary_chunks, NATURAL_TYPE secondary_chunk_size, NATURAL_TYPE secondary_slave_size, NATURAL_TYPE secondary_size, NATURAL_TYPE secondary_num_slaves){
	NATURAL_TYPE i, j, k, start;
	int slave_id;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &slave_id);
	slave_id = slave_id - 1;
	
	start = (recive_index * secondary_slave_size) - ((slave_id << (log2_64((uint64_t) result_vector->assigned_elements))) % (secondary_num_slaves * secondary_slave_size));
	for(i = 0; i < num_secondary_chunks; i++){
		MPI_Recv(recived_secondary_chunk, secondary_chunk_size, MPI_DOUBLE_COMPLEX, recive_index+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // #POK			
		for(j = secondary_chunk_size*i; j < result_vector->assigned_elements; j=j+secondary_size){
			for(k = 0; k < secondary_chunk_size; k++){
				result_vector->vector[start+j+k] = COMPLEX_MULT(result_vector->vector[start+j+k], recived_secondary_chunk[k]);
			}
		}
	}
}

// Revisada // TODO: VER SI EN LUGAR DE USAR local_state_vectors[INDEX] SE USA DIRECTAMENTE EL CAST
unsigned char slave_state_join(struct slave_state_vector **local_state_vectors) {
	NATURAL_TYPE num_secondary_chunks, num_main_chunks, main_slave_size, secondary_slave_size;
	NATURAL_TYPE i, main_chunk_size, secondary_chunk_size, secondary_size;
	int mtrsr, slave_id, response, communication_index, result_index, main_index, secondary_index;
	int secondary_num_slaves, main_num_slaves, value, errored;
	COMPLEX_TYPE *recived_main_chunk = NULL, *recived_secondary_chunk = NULL;
	struct slave_state_vector *result_vector, *main_vector, *secondary_vector;
	int parameters1[6];
	NATURAL_TYPE parameters2[7];
	bool recive, send;
	// Parametros MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &slave_id);
	slave_id = slave_id - 1;
	errored = 0;
	
	// Recibiendo parametros necesarios	
	MPI_Recv(parameters1, 6, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	unpack_int(parameters1, 6, &main_index, &secondary_index, &result_index, &mtrsr, &secondary_num_slaves, &main_num_slaves);
	MPI_Recv(parameters2, 7, MPI_NATURAL_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	unpack_natural_type(parameters2, 7, &secondary_size, &main_slave_size, &secondary_slave_size, &main_chunk_size, &secondary_chunk_size, &num_secondary_chunks, &num_main_chunks);

	// Allocate send_main_chunk, recived_main_chunk and recived_secondary_chunk if needed
	SAFE_MALLOC_CONDITIONAL(recived_main_chunk, main_chunk_size, COMPLEX_TYPE, errored, 3, cleanup, (mtrsr > 1));
	SAFE_MALLOC_CONDITIONAL(recived_secondary_chunk, secondary_chunk_size, COMPLEX_TYPE, errored, 3, cleanup, (mtrsr != 1 || main_num_slaves != 1 || secondary_num_slaves != 1));
	MPI_NOTIFY_READY(response, cleanup);
	
	// ENVIA LOS ELEMENTOS DE MAIN VECTOR
	if (slave_id < main_num_slaves){
		for(i = 0; i < local_state_vectors[result_index]->num_slaves; i++){
			communication_index = slave_id * mtrsr + i;
			if (communication_index == slave_id){
				join_process_self_main(local_state_vectors[result_index], local_state_vectors[main_index], main_slave_size);
			} else if ((communication_index / mtrsr) == slave_id) {
				join_send_main(local_state_vectors[main_index], communication_index, mtrsr, num_main_chunks, main_chunk_size);
			}
		}
	}
	
	// RECIBE LOS ELEMENTOS DE MAIN VECTOR
	if (slave_id > 0 && mtrsr != 1){
		join_recive_main(local_state_vectors[result_index], recived_main_chunk, main_chunk_size, main_slave_size, (slave_id / mtrsr), num_main_chunks);
	}
	
	// ENVIA, RECIBE Y PROCESA SECONDARY VECTOR
	for(i = 0; i < local_state_vectors[result_index]->num_slaves; i++){
		communication_index = (i - slave_id + local_state_vectors[result_index]->num_slaves) % local_state_vectors[result_index]->num_slaves;
		value = (secondary_slave_size * secondary_num_slaves) >> log2_64((uint64_t) local_state_vectors[result_index]->assigned_elements);
		if (value == 0){
			recive = communication_index < secondary_num_slaves;
			send = slave_id < secondary_num_slaves;
		} else {
			recive = (slave_id % value) == ((communication_index * secondary_slave_size)/local_state_vectors[result_index]->assigned_elements);
			send = (communication_index % value) == ((slave_id * secondary_slave_size)/local_state_vectors[result_index]->assigned_elements);
		}
		if (slave_id == communication_index && slave_id < secondary_num_slaves){
			join_process_self_secondary(local_state_vectors[result_index], local_state_vectors[secondary_index]);
		}
		if (send && slave_id < communication_index && slave_id != communication_index) {
			join_send_secondary(local_state_vectors[secondary_index], communication_index, num_secondary_chunks, secondary_chunk_size, secondary_num_slaves);
		}
		if (recive && slave_id != communication_index){
			join_recive_secondary(local_state_vectors[result_index], recived_secondary_chunk, communication_index, num_secondary_chunks, secondary_chunk_size, secondary_slave_size, secondary_size, secondary_num_slaves);
		}
		if (send && slave_id >= communication_index && slave_id != communication_index) {
			join_send_secondary(local_state_vectors[secondary_index], communication_index, num_secondary_chunks, secondary_chunk_size, secondary_num_slaves);
		}
	}
	
	MPI_NOTIFY_FINISHED(response);
	
cleanup:
	SAFE_FREE(recived_main_chunk);
	SAFE_FREE(recived_secondary_chunk);
	
	if (errored){
		MPI_SEND_ABORT(response);
		return errored;
	}
	return 0;
}

// Revisada
unsigned char master_state_join(struct master_state_vector *result_vector, struct master_state_vector *main_vector, struct master_state_vector *secondary_vector) {
	NATURAL_TYPE secondary_size, main_slave_size, secondary_slave_size, main_chunk_size, secondary_chunk_size, num_main_chunks;
	unsigned char exit_code;
	int i, response, errored, result_index, main_index, secondary_index, mpi_size;
	int mtrsr, secondary_num_slaves, main_num_slaves, num_secondary_chunks;
	
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	exit_code = master_state_init(result_vector, main_vector->num_qubits + secondary_vector->num_qubits, false, mpi_size-1);
	if (exit_code != 0) {
		return exit_code;
	}

	// Calcula todos los parametros simples
	secondary_num_slaves = secondary_vector->num_slaves;
	main_num_slaves = main_vector->num_slaves;
	secondary_size = secondary_vector->size;
	mtrsr = result_vector->num_slaves / main_vector->num_slaves;
	main_slave_size = MAX(1, (main_vector->size / main_vector->num_slaves) / mtrsr);
	secondary_slave_size = secondary_size / secondary_num_slaves;
	main_chunk_size = MIN(SLAVE_COMMUNICATION_CHUNK_SIZE, main_slave_size);
	secondary_chunk_size = MIN(SLAVE_COMMUNICATION_CHUNK_SIZE, secondary_slave_size);
	num_main_chunks = main_slave_size / main_chunk_size;
	num_secondary_chunks = secondary_slave_size / secondary_chunk_size;

	// Envia el comando para iniciar slave_state_join y los parametros previamente calculados
	send_slaves_ints(result_vector, 2, 10, 0);
	for (i = 0; i < result_vector->num_slaves; i++){
		main_index = (i < main_vector->num_slaves) ? main_vector->slave_ids[i] : -1;
		secondary_index = (i < secondary_vector->num_slaves) ? secondary_vector->slave_ids[i] : -1;
		result_index = result_vector->slave_ids[i];
		send_ints(i+1, 6, main_index, secondary_index, result_index, mtrsr, secondary_num_slaves, main_num_slaves);
		
	}
	send_slaves_natural_types(result_vector, 7, secondary_size, main_slave_size, secondary_slave_size, main_chunk_size, secondary_chunk_size, num_secondary_chunks, num_main_chunks);
	
	// Gestiona los errores de los mallocs y envia confirmacion positiva o negativa a los esclavos
	recive_confirmations(result_vector, &errored);  // #AWC
	send_slaves_ints(result_vector, 1, !errored);  // #VGW
	
	// Espera a que terminen todos
	printf("MASTER esperando confirmaciones finales\n");
	recive_confirmations(result_vector, &errored);  // #JFM
	return 0;
}

// Revisada
void slave_probability_measure_one(struct slave_state_vector **local_state_vectors) {
	NATURAL_TYPE i, element_index, num_elements, low, high, target_mask;
	REAL_TYPE sq_sum;
	COMPLEX_TYPE element;
	int parameters[2];
	int msb_bitsize, lsb_bitsize, state_index, target_index, slave_id;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &slave_id);
	slave_id = slave_id - 1;
	
	MPI_Recv(parameters, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // #DDD
	unpack_int(parameters, 2, &target_index, &state_index);

	
	target_mask = NATURAL_ONE << target_index;
	msb_bitsize = ffs(local_state_vectors[state_index]->num_slaves) - 1;
	lsb_bitsize = local_state_vectors[state_index]->num_qubits - msb_bitsize;
	low = target_mask - 1;
	high = ~low;
	
	if (target_index < lsb_bitsize){
		// La mitad de los elementos cumplen la condicion
		for (i = 0; i < (local_state_vectors[state_index]->assigned_elements >> 1); i++) {
			element_index = ((i & high) << 1) + target_mask + (i & low);
			element = local_state_vectors[state_index]->vector[element_index];
			sq_sum += RE(element) * RE(element) + IM(element) * IM(element);
		}
	} else if ((slave_id << lsb_bitsize) & target_mask) {
		// Todos los elementos cumplen la condicion
		for (element_index = 0; element_index < local_state_vectors[state_index]->assigned_elements; element_index++) {
			element = local_state_vectors[state_index]->vector[element_index];
			sq_sum += RE(element) * RE(element) + IM(element) * IM(element);
		}
	}
	MPI_Send(&sq_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // #EEE
}

// Revisada
REAL_TYPE master_probability_measure_one(struct master_state_vector *master_state, unsigned int target_index) {
	int request1[2];
	int i; 
	unsigned int request2[2];
	REAL_TYPE value, sq_sum;
	
	send_slaves_ints(master_state, 2, 5, 0);
	
	request1[0] = (int) target_index;
	for (i = 0; i<master_state->num_slaves; i++){
		request1[1] = (int) master_state->slave_ids[i];
		MPI_Send(request1, 2, MPI_INT, i+1, 0, MPI_COMM_WORLD); // #DDD
	}
	
	sq_sum = 0;
	for (i = 0; i<master_state->num_slaves; i++){
		MPI_Recv(&value, 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // #EEE
		sq_sum += value;
	}
	
	return sq_sum;
}


// Revisada
unsigned char measure_master_state_vector(struct master_state_vector *master_state, bool *measure, unsigned int target_index, struct master_state_vector *new_master_state, REAL_TYPE roll) {
	REAL_TYPE sq_sum;
	unsigned char exit_code;
	sq_sum = master_probability_measure_one(master_state, target_index);
	*measure = sq_sum > roll;
	exit_code = collapse_master_state(master_state, target_index, *measure, new_master_state);
	return exit_code;
}

// Revisada
unsigned char slave_recive_collapsed_state_vector(struct slave_state_vector **local_state_vectors, int new_state_index, NATURAL_TYPE recive_length, int slave_to_recive, NATURAL_TYPE recive_offset){
	NATURAL_TYPE i, chunk_size, chunks_to_recive;
	
	chunk_size = MIN(recive_length, SLAVE_COLLAPSE_CHUNK_SIZE);
	chunks_to_recive = recive_length / chunk_size;

	for (i = 0; i < chunks_to_recive; i++) {
		MPI_Recv(&local_state_vectors[new_state_index]->vector[recive_offset+i*chunk_size], chunk_size, MPI_DOUBLE_COMPLEX, slave_to_recive+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // #ABC
	}
}

// Revisada
unsigned char slave_send_collapsed_state_vector(struct slave_state_vector **local_state_vectors, int old_state_index, int value, int target_index, NATURAL_TYPE send_length, int slave_to_send, NATURAL_TYPE send_offset){
	NATURAL_TYPE i, next_index, chunk_size, chunks_to_send, block_size, low_mask, high_mask, value_mask, all_mask, offset, j, k;
	COMPLEX_TYPE *chunk_to_send = NULL;
	int errored;
	
	chunk_size = MIN(send_length, SLAVE_COLLAPSE_CHUNK_SIZE);
	chunks_to_send = send_length / chunk_size;
	block_size = NATURAL_ONE << target_index;
	low_mask = block_size - 1;
	high_mask = ~low_mask;
	value_mask = block_size * value;
	all_mask = local_state_vectors[old_state_index]->assigned_elements - 1;
	offset = send_offset;
	
	SAFE_MALLOC(chunk_to_send, chunk_size, COMPLEX_TYPE, errored, 1, cleanup);
	
	for (j = 0; j < chunks_to_send; j++) {
		for (k = 0; k < chunk_size; k++) {
			i = offset + k;
			next_index = (((i & high_mask) << 1) + value_mask + (i & low_mask)) & all_mask;
			chunk_to_send[k] = local_state_vectors[old_state_index]->vector[next_index];
		}
		offset += chunk_size;
		MPI_Send(chunk_to_send, chunk_size, MPI_DOUBLE_COMPLEX, slave_to_send+1, 0, MPI_COMM_WORLD); //#ABC
	}
	
cleanup:	
	SAFE_FREE(chunk_to_send);
	return 0;
}

// Revisada
void slave_send_self_state_vector(struct slave_state_vector **local_state_vectors, int old_state_index, int new_state_index, int value, int target_index, NATURAL_TYPE comm_length, NATURAL_TYPE send_offset){
	NATURAL_TYPE i, index, low_mask, high_mask, value_mask, all_mask;
	
	low_mask = (NATURAL_ONE << target_index) - 1;
	high_mask = ~low_mask;
	value_mask = (NATURAL_ONE << target_index) * value;
	all_mask = local_state_vectors[old_state_index]->assigned_elements-1;
	
	for (i = 0; i < comm_length; i++) {
		index = (((((i & high_mask) << 1) | value_mask | (i & low_mask))) & all_mask) + send_offset;;
		local_state_vectors[new_state_index]->vector[i] = local_state_vectors[old_state_index]->vector[index];
	}
}


// Revisada
unsigned char collapse_slave_state(struct slave_state_vector **local_state_vectors) {
	int old_state_index, new_state_index, value, target_index, msb_target_index, new_state_slaves, lsb_bitsize;
	int comm_id, comm_size, send_offset, recv_offset, is_target_in_lsb, slaves_are_reduced, slave_id;
	int i, a, br, cr, dr, be, ce, de, te, tr;
	int parameters[5];

	// MPI PARAMETERS
	MPI_Comm_rank(MPI_COMM_WORLD, &slave_id);
	slave_id = slave_id - 1;
	
	MPI_Recv(parameters, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // #III
	unpack_int(parameters, 5, &old_state_index, &new_state_index, &value, &target_index, &new_state_slaves);
	
	lsb_bitsize = local_state_vectors[old_state_index]->num_qubits - (ffs(local_state_vectors[old_state_index]->num_slaves) - 1);
	msb_target_index = target_index - lsb_bitsize;
	is_target_in_lsb = target_index < lsb_bitsize;
	slaves_are_reduced = local_state_vectors[old_state_index]->num_slaves > new_state_slaves;
	te = ((target_index < lsb_bitsize) || (((slave_id >> msb_target_index) & 1) == value));
	tr = (slave_id < new_state_slaves);
	comm_size = ((!is_target_in_lsb&&slaves_are_reduced) ? local_state_vectors[old_state_index]->assigned_elements : local_state_vectors[old_state_index]->assigned_elements >> 1);
	for (i = 0; i<local_state_vectors[old_state_index]->num_slaves; i++){
		comm_id = (i - slave_id + local_state_vectors[old_state_index]->num_slaves) % local_state_vectors[old_state_index]->num_slaves;
		a = (slave_id == comm_id);
		if (te){
			be = (COMPARE_BIT_REMOVAL(slave_id, comm_id, msb_target_index));
			ce = ((slave_id >> 1) == comm_id);
			de = (REMOVE_BIT_AND_SHIFT_HIGH(slave_id, msb_target_index) == comm_id);
			if (slaves_are_reduced ? (is_target_in_lsb ? ce : de) : (is_target_in_lsb ? a : be)){
				if ((comm_id % 2) && !slaves_are_reduced && !is_target_in_lsb){
					send_offset = local_state_vectors[old_state_index]->assigned_elements >> 1;
				} else {
					send_offset = 0;	
				}
				if (a){
					recv_offset = ((comm_id % 2) && slaves_are_reduced && is_target_in_lsb) ? (local_state_vectors[old_state_index]->assigned_elements >> 1) : 0;
					slave_send_self_state_vector(local_state_vectors, old_state_index, new_state_index, value, target_index, comm_size, send_offset);
				} else {
					slave_send_collapsed_state_vector(local_state_vectors, old_state_index, value, target_index, comm_size, comm_id, send_offset);
				}
			}
		}
		if (tr && !a){
			br = (INSERT_BIT_AT_INDEX_SHIFTRIGHT(slave_id, msb_target_index, value) == comm_id);
			cr = ((comm_id >> 1) == slave_id);
			dr = (INSERT_BIT_AT_INDEX_SHIFTLEFT(slave_id, msb_target_index, value) == comm_id);
			if (slaves_are_reduced ? (is_target_in_lsb ? cr : dr) : (!is_target_in_lsb && br)){
				recv_offset = ((comm_id % 2) && slaves_are_reduced && is_target_in_lsb) ? (local_state_vectors[old_state_index]->assigned_elements >> 1) : 0;
				slave_recive_collapsed_state_vector(local_state_vectors,  new_state_index, comm_size, comm_id, recv_offset);
			}
		}
	}
	return 0;
}

// Revisada
unsigned char collapse_master_state(struct master_state_vector *master_state, unsigned int target_index, int value, struct master_state_vector *new_master_state) {
	unsigned char exit_code;
	int parameters[5];
	int mpi_size, i, errored = 0;

	if (master_state->num_qubits == 1) {
		send_ints(1, 2, 2, (int) master_state->slave_ids[0]);
		new_master_state->slave_ids = NULL;
		new_master_state->num_qubits = 0;
		new_master_state->size = 0;
		new_master_state->num_slaves = 0;
		return 0;
	}
	
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	exit_code =  master_state_init(new_master_state, master_state->num_qubits - 1, 0, mpi_size-1);
	if (exit_code) {
		errored = (int) exit_code;
		goto cleanup;
	}
	
	send_slaves_ints(master_state, 2, 7, 0);

	parameters[2] = value;
	parameters[3] = (int) target_index;
	parameters[4] = (int) new_master_state->num_slaves;
	for(i = 0; i < master_state->num_slaves; i++){
		parameters[0] = (int) master_state->slave_ids[i];
		parameters[1] = (i < new_master_state->num_slaves) ? (int)new_master_state->slave_ids[i] : -1;
		MPI_Send(parameters, 5, MPI_INT, i+1, 0, MPI_COMM_WORLD); // #III
	}
	
	
	exit_code = normalize_master_state(new_master_state);
	if (exit_code){
		errored = 3;
	}

cleanup:
	if (errored){
		master_state_free(new_master_state);
		return (unsigned char) errored;
	}
	return 0;
}


REAL_TYPE get_global_phase(struct state_vector *state) {
	NATURAL_TYPE i;
	REAL_TYPE phase;
	COMPLEX_TYPE val;

	if (state->fcarg_init) {
		return state->fcarg;
	}

	phase = 0.0;
	for (i = 0; i < state->size; i++) {
		val = state_get(state, i);
		if (RE(val) != 0. || IM(val) != 0.) {
			if (IM(val) != 0.) {
				phase = ARG(val);
			}
			break;
		}
	}
	state->fcarg = phase;
	state->fcarg_init = 1;

	return phase;
}

REAL_TYPE probability(struct state_vector *state, unsigned int target_id) {
	NATURAL_TYPE i, index, qty, low, high, target;
	REAL_TYPE value;
	COMPLEX_TYPE val;

	qty = state->size >> 1;
	target = NATURAL_ONE << target_id;
	low = target - 1;
	high = ~low;

	value = 0;
#pragma omp parallel for reduction (+:value) default (none) firstprivate (state, qty, low, high, target, COMPLEX_ARRAY_SIZE) private (i, index, val)
	for (i = 0; i < qty; i++) {
		index = ((i & high) << 1) + target + (i & low);
		val = state_get(state, index);
		value += RE(val) * RE(val) + IM(val) * IM(val);
	}

	return value;
}

unsigned char join(struct state_vector *r, struct state_vector *s1, struct state_vector *s2) {
	NATURAL_TYPE i, j, new_index;
	COMPLEX_TYPE o1, o2;
	unsigned char exit_code;

	exit_code = state_init(r, s1->num_qubits + s2->num_qubits, false);
	if (exit_code != 0) {
		return exit_code;
	}

#pragma omp parallel for default(none) firstprivate(r, s1, s2, exit_code, COMPLEX_ARRAY_SIZE) private(i, j, o1, o2, new_index)
	for (i = 0; i < s1->size; i++) {
		o1 = state_get(s1, i);
		for (j = 0; j < s2->size; j++) {
			new_index = i * s2->size + j;
			o2 = state_get(s2, j);
			state_set(r, new_index, COMPLEX_MULT(o1, o2));
		}
	}

	return 0;
}

unsigned char measure(struct state_vector *state, bool *result, unsigned int target, struct state_vector *new_state, REAL_TYPE roll) {
	REAL_TYPE sum;
	unsigned char exit_code;

	sum = probability(state, target);
	*result = sum > roll;
	exit_code = collapse(state, target, *result, sum, new_state);

	return exit_code;
}

unsigned char collapse(struct state_vector *state, unsigned int target_id, bool value, REAL_TYPE prob_one, struct state_vector *new_state) {
	unsigned char exit_code;
	NATURAL_TYPE i, j, low, high, val;

	if (state->num_qubits == 1) {
		new_state->vector = NULL;
		new_state->num_qubits = 0;
		return 0;
	}

	exit_code = state_init(new_state, state->num_qubits - 1, false);
	if (exit_code != 0) {
		free(new_state);
		return exit_code;
	}
	val = NATURAL_ONE << target_id;
	low = val - 1;
	high = ~low;
	if (!value) {
		prob_one = 1 - prob_one;
		val = 0;
	}

#pragma omp parallel for default(none) firstprivate(state, new_state, low, high, val, COMPLEX_ARRAY_SIZE) private(i, j)
	for (j = 0; j < new_state->size; j++) {
		i = ((j & high) << 1) + val + (j & low);
		state_set(new_state, j, state_get(state, i));
	}
	new_state->norm_const = sqrt(prob_one);

	return 0;
}


unsigned char apply_gate(struct state_vector *state, struct qgate *gate, unsigned int *targets, unsigned int num_targets, unsigned int *controls, unsigned int num_controls, unsigned int *anticontrols, unsigned int num_anticontrols, struct state_vector *new_state) {
	REAL_TYPE norm_const;
	unsigned char exit_code;
	NATURAL_TYPE control_mask, anticontrol_mask, i, reg_index;
	unsigned int j, k, row;
	COMPLEX_TYPE sum;

	if (new_state == NULL)
		return 10;

	exit_code = state_init(new_state, state->num_qubits, false);
	// 0 -> OK
	// 1 -> Error initializing chunk
	// 2 -> Error allocating chunk
	// 3 -> Error setting values (should never happens since init = 0)
	// 4 -> Error allocating state
	if (exit_code != 0) {
		free(new_state);
		return exit_code;
	}

	control_mask = NATURAL_ZERO;
	for (j = 0; j < num_controls; j++)
		control_mask |= NATURAL_ONE << controls[j];
	anticontrol_mask = NATURAL_ZERO;
	for (j = 0; j < num_anticontrols; j++)
		anticontrol_mask |= NATURAL_ONE << anticontrols[j];

	norm_const = 0;
#pragma omp parallel for reduction (+:norm_const) \
                                     default(none) \
                                     firstprivate (state, new_state, gate, \
                        			   targets, num_targets, \
                        			   controls, num_controls, \
                        			   anticontrols, num_anticontrols, \
                        			   control_mask, anticontrol_mask, \
                        			   COMPLEX_ZERO, COMPLEX_ARRAY_SIZE) \
                                     private (sum, row, reg_index, i, j, k)
	for (i = 0; i < state->size; i++) {
		if ((i & control_mask) == control_mask && (i & anticontrol_mask) == 0) {
			// Calculate
			sum = COMPLEX_ZERO;
			reg_index = i;
			// We have gate->size elements to add in sum
			for (j = 0; j < gate->size; j++) {
				// We get the value of each target qubit id on the current new state
				// element and we store it in rowbits following the same order as the
				// one in targets
				row = 0;
				for (k = 0; k < num_targets; k++) {
					row += ((i & (NATURAL_ONE << targets[k])) != 0) << k;
					// We check the value of the kth bit of j
					// and set the value of the kth target bit to it
					if ((j & (NATURAL_ONE << k)) != 0){
						reg_index |= NATURAL_ONE << targets[k];
					} else {
						reg_index &= ~(NATURAL_ONE << targets[k]);
					}
				}
				sum = COMPLEX_ADD(sum, COMPLEX_MULT(state_get(state, reg_index), gate->matrix[row][j]));
			}
		} else {
			//Copy
			sum = state_get(state, i);
		}
		state_set(new_state, i, sum);
		norm_const += pow(RE(sum), 2) + pow(IM(sum), 2);
	}
	new_state->norm_const = sqrt(norm_const);

	return 0;
}

#ifndef _MSC_VER
__attribute__((const))
#endif
static COMPLEX_TYPE
_densityFun(NATURAL_TYPE i, NATURAL_TYPE j,
#ifndef _MSC_VER
	    NATURAL_TYPE unused1 __attribute__((unused)),
	    NATURAL_TYPE unused2 __attribute__((unused)),
#else
	    NATURAL_TYPE unused1, NATURAL_TYPE unused2,
#endif
	    void *rawstate)
{
	COMPLEX_TYPE elem_i, elem_j, result;
	struct state_vector *state =
		(struct state_vector *)PyCapsule_GetPointer(
			rawstate, "qsimov.doki.state_vector");
	if (state == NULL) {
		return COMPLEX_NAN;
	}

	elem_i = state_get(state, i);
	elem_j = state_get(state, j);
	// printf("state[" NATURAL_STRING_FORMAT "] = " COMPLEX_STRING_FORMAT "\n", i,
	// COMPLEX_STRING(elem_i)); printf("state[" NATURAL_STRING_FORMAT "] = " COMPLEX_STRING_FORMAT
	// "\n", j, COMPLEX_STRING(elem_i));
	result = COMPLEX_MULT(elem_i, conj(elem_j));
	// printf("result = " COMPLEX_STRING_FORMAT "\n", COMPLEX_STRING(result));

	return result;
}

struct Application {
	/* Capsule containing the state */
	PyObject *state_capsule;
	/* State vector */
	struct FMatrix *state;
	/* Capsule containing the gate */
	PyObject *gate_capsule;
	/* Gate matrix */
	struct FMatrix *gate;
	/* Target qubit indexes */
	unsigned int *targets;
	/* Control qubit indexes */
	unsigned int *controls;
	/* Anticontrol qubit indexes */
	unsigned int *anticontrols;
	/* How many references are there to this object */
	NATURAL_TYPE refcount;
	/* Number of target qubits */
	unsigned int num_targets;
	/* Number of control qubits */
	unsigned int num_controls;
	/* Number of anticontrol qubits */
	unsigned int num_anticontrols;
};

static struct Application *
new_application(PyObject *state_capsule, PyObject *gate_capsule,
		unsigned int *targets, unsigned int num_targets,
		unsigned int *controls, unsigned int num_controls,
		unsigned int *anticontrols, unsigned int num_anticontrols);

static void free_application(void *raw_app);

static void *clone_application(void *raw_app);

static size_t size_application(void *raw_app);

#ifndef _MSC_VER
__attribute__((const))
#endif
static COMPLEX_TYPE
_ApplyGateFunction(NATURAL_TYPE i,
#ifndef _MSC_VER
		   NATURAL_TYPE unused1 __attribute__((unused)),
		   NATURAL_TYPE unused2 __attribute__((unused)),
		   NATURAL_TYPE unused3 __attribute__((unused)),
#else
		   NATURAL_TYPE unused1, NATURAL_TYPE unused2,
		   NATURAL_TYPE unused3,
#endif
		   void *raw_app);

static struct Application *
new_application(PyObject *state_capsule, PyObject *gate_capsule,
		unsigned int *targets, unsigned int num_targets,
		unsigned int *controls, unsigned int num_controls,
		unsigned int *anticontrols, unsigned int num_anticontrols)
{
	struct Application *data = MALLOC_TYPE(1, struct Application);

	if (data != NULL) {
		struct FMatrix *state, *gate;

		state = (struct FMatrix *)PyCapsule_GetPointer(
			state_capsule, "qsimov.doki.funmatrix");
		gate = (struct FMatrix *)PyCapsule_GetPointer(
			gate_capsule, "qsimov.doki.funmatrix");

		if (state == NULL) {
			errno = 4;
			return NULL;
		}
		if (gate == NULL) {
			errno = 3;
			return NULL;
		}

		Py_INCREF(state_capsule);
		data->state_capsule = state_capsule;
		data->state = state;
		Py_INCREF(gate_capsule);
		data->gate_capsule = gate_capsule;
		data->gate = gate;
		data->targets = targets;
		data->num_targets = num_targets;
		data->controls = controls;
		data->num_controls = num_controls;
		data->anticontrols = anticontrols;
		data->num_anticontrols = num_anticontrols;
		data->refcount = 1;
	}

	return data;
}

static void free_application(void *raw_app) {
	struct Application *data = (struct Application *)raw_app;

	if (data == NULL) {
		return;
	}

	data->refcount--;
	if (data->refcount == 0) {
		Py_DECREF(data->state_capsule);
		data->state_capsule = NULL;
		data->state = NULL;
		Py_DECREF(data->gate_capsule);
		data->gate_capsule = NULL;
		data->gate = NULL;
		free(data->targets);
		data->targets = NULL;
		free(data->controls);
		data->controls = NULL;
		free(data->anticontrols);
		data->anticontrols = NULL;
		data->num_targets = 0;
		data->num_controls = 0;
		data->num_anticontrols = 0;
		free(data);
	}
}

static void *clone_application(void *raw_app) {
	struct Application *data = (struct Application *)raw_app;

	if (data == NULL) {
		return NULL;
	}

	data->refcount++;
	return raw_app;
}

static size_t size_application(void *raw_app) {
	size_t size;
	struct Application *data = (struct Application *)raw_app;

	if (data == NULL) {
		return 0;
	}
	size = sizeof(struct Application);
	size += FM_mem_size(data->state);
	size += FM_mem_size(data->gate);
	size += data->num_targets * sizeof(unsigned int);
	size += data->num_controls * sizeof(unsigned int);
	size += data->num_anticontrols * sizeof(unsigned int);

	return size;
}

struct FMatrix *apply_gate_fmat(PyObject *state_capsule, PyObject *gate_capsule,
				unsigned int *targets, unsigned int num_targets,
				unsigned int *controls,
				unsigned int num_controls,
				unsigned int *anticontrols,
				unsigned int num_anticontrols)
{
	struct FMatrix *pFM;
	struct Application *data = new_application(
		state_capsule, gate_capsule, targets, num_targets, controls,
		num_controls, anticontrols, num_anticontrols);

	if (data == NULL) {
		errno = 5;
		return NULL;
	}

	pFM = new_FunctionalMatrix(data->state->r, 1, &_ApplyGateFunction, data,
				   free_application, clone_application,
				   size_application);
	if (pFM == NULL) {
		errno = 1;
		free_application(data);
	}

	return pFM;
}

#ifndef _MSC_VER
__attribute__((const))
#endif
static COMPLEX_TYPE
_ApplyGateFunction(NATURAL_TYPE i,
#ifndef _MSC_VER
		   NATURAL_TYPE unused1 __attribute__((unused)),
		   NATURAL_TYPE unused2 __attribute__((unused)),
		   NATURAL_TYPE unused3 __attribute__((unused)),
#else
		   NATURAL_TYPE unused1, NATURAL_TYPE unused2,
		   NATURAL_TYPE unused3,
#endif
		   void *raw_app)
{
	int res;
	NATURAL_TYPE k;
	long long n;
	NATURAL_TYPE mask, row, reg_index = i;
	COMPLEX_TYPE val = COMPLEX_ZERO;
	struct Application *data = (struct Application *)raw_app;

	for (k = 0; k < data->num_controls; ++k) {
		mask = NATURAL_ONE << data->controls[k];
		if (!(i & mask)) {
			res = getitem(data->state, i, 0, &val);
			if (res != 0) {
				printf("Error[C] %d while getting state item " NATURAL_STRING_FORMAT
				       "\n",
				       res, i);
				return COMPLEX_NAN;
			}
			return val;
		}
	}

	for (k = 0; k < data->num_anticontrols; ++k) {
		mask = NATURAL_ONE << data->anticontrols[k];
		if (i & mask) {
			res = getitem(data->state, i, 0, &val);
			if (res != 0) {
				printf("Error[A] %d while getting state item " NATURAL_STRING_FORMAT
				       "\n",
				       res, i);
				return COMPLEX_NAN;
			}
			return val;
		}
	}

	for (n = 0; n < data->gate->r; ++n) {
		// We get the value of each target qubit id on the current new state
		// element and we store it in rowbits following the same order as the
		// one in targets
		COMPLEX_TYPE aux, aux2;

		row = 0;
		for (k = 0; k < data->num_targets; k++) {
			row += ((i & (NATURAL_ONE << data->targets[k])) != 0)
			       << k;
			// We check the value of the kth bit of j
			// and set the value of the kth target bit to it
			if ((n & (NATURAL_ONE << k)) != 0) {
				reg_index |= NATURAL_ONE << data->targets[k];
			} else {
				reg_index &= ~(NATURAL_ONE << data->targets[k]);
			}
		}
		res = getitem(data->state, reg_index, 0, &aux);
		if (res != 0) {
			printf("Error[T] %d while getting state[" NATURAL_STRING_FORMAT
			       "] item " NATURAL_STRING_FORMAT "\n",
			       res, i, reg_index);
			return COMPLEX_NAN;
		}
		res = getitem(data->gate, row, n, &aux2);
		if (res != 0) {
			printf("Error[T] %d while getting gate item " NATURAL_STRING_FORMAT
			       ", " NATURAL_STRING_FORMAT "\n",
			       res, row, n);
			return COMPLEX_NAN;
		}
		val = COMPLEX_ADD(val, COMPLEX_MULT(aux, aux2));
	}

	return val;
}

static size_t size_state_capsule(void *raw_capsule)
{
	struct state_vector *state;
	PyObject *capsule = (PyObject *)raw_capsule;

	if (capsule == NULL) {
		return 0;
	}

	state = (struct state_vector *)PyCapsule_GetPointer(
		capsule, "qsimov.doki.state_vector");

	return state_mem_size(state);
}

struct FMatrix *density_matrix(PyObject *state_capsule) {
	struct FMatrix *dm = NULL;
	struct state_vector *state =
		PyCapsule_GetPointer(state_capsule, "qsimov.doki.state_vector");

	if (state != NULL) {
		dm = new_FunctionalMatrix(state->size, state->size,
					  &_densityFun, state_capsule,
					  free_capsule, clone_capsule,
					  size_state_capsule);
		if (dm != NULL) {
			Py_INCREF(state_capsule);
		} else {
			errno = 1;
		}
	} else {
		errno = 2;
	}

	return dm;
}

#ifndef _MSC_VER
__attribute__((const))
#endif
static COMPLEX_TYPE
_densityFun(NATURAL_TYPE i, NATURAL_TYPE j,
#ifndef _MSC_VER
	    NATURAL_TYPE unused1 __attribute__((unused)),
	    NATURAL_TYPE unused2 __attribute__((unused)),
#else
	    NATURAL_TYPE unused1, NATURAL_TYPE unused2,
#endif
	    void *rawstate);