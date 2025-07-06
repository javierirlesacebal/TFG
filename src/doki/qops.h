/*
 * Doki: Quantum Computer simulator, using state vectors. QSimov core.
 * Copyright (C) 2021  Hernán Indíbil de la Cruz Calvo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once
#ifndef QOPS_H_
#define QOPS_H_
#include "funmatrix.h"
#include "qgate.h"
#include "qstate.h"
#include <Python.h>


unsigned char join(struct state_vector *r, struct state_vector *s1, struct state_vector *s2);

unsigned char measure(struct state_vector *state, bool *result, unsigned int target, struct state_vector *new_state, REAL_TYPE roll);

REAL_TYPE probability(struct state_vector *state, unsigned int target_id);

REAL_TYPE get_global_phase(struct state_vector *state);

unsigned char collapse(struct state_vector *state, unsigned int id, bool value, REAL_TYPE prob_one, struct state_vector *new_state);

unsigned char apply_gate(struct state_vector *state, struct qgate *gate,
			 unsigned int *targets, unsigned int num_targets,
			 unsigned int *controls, unsigned int num_controls,
			 unsigned int *anticontrols,
			 unsigned int num_anticontrols,
			 struct state_vector *new_state);

struct FMatrix *apply_gate_fmat(PyObject *state_capsule, PyObject *gate_capsule,
				unsigned int *targets, unsigned int num_targets,
				unsigned int *controls,
				unsigned int num_controls,
				unsigned int *anticontrols,
				unsigned int num_anticontrols);

struct FMatrix *density_matrix(PyObject *state_capsule);

// Fase Global
REAL_TYPE get_master_global_phase(struct master_state_vector *master_state);

// Normalizar
void slave_state_normalize(struct slave_state_vector **local_state_vectors, int state_index);
void normalize_master_state(struct master_state_vector *master_state);

// Apply gate
unsigned char apply_gate_to_master_state(struct master_state_vector *master_state, struct master_state_vector *new_master_state, int num_targets, unsigned int *targets, unsigned int num_controls,unsigned int *controls, unsigned int num_anticontrols, unsigned int *anticontrols);
unsigned char apply_gate_to_slave_state(struct slave_state_vector **local_state_vectors, struct qgate *gate);

// Join / Producto tensorial
void join_process_self_main(struct slave_state_vector *result_vector, struct slave_state_vector *main_vector, NATURAL_TYPE main_slave_size);
void join_send_main(struct slave_state_vector *main_vector, int send_index, int mtrsr, NATURAL_TYPE num_chunks, NATURAL_TYPE chunk_size);
void join_recive_main(struct slave_state_vector *result_vector, COMPLEX_TYPE *recived_main_chunk, NATURAL_TYPE main_chunk_size, NATURAL_TYPE main_slave_size, int recive_from, NATURAL_TYPE num_main_chunks);
void join_process_self_secondary(struct slave_state_vector *result, struct slave_state_vector *secondary_vector);
void join_send_secondary(struct slave_state_vector *secondary, int send_index, NATURAL_TYPE num_secondary_chunks, NATURAL_TYPE secondary_chunk_size, NATURAL_TYPE secondary_num_slaves);
void join_recive_secondary(struct slave_state_vector *result_vector, COMPLEX_TYPE * recived_secondary_chunk, int recive_index, NATURAL_TYPE num_secondary_chunks, NATURAL_TYPE secondary_chunk_size, NATURAL_TYPE secondary_slave_size, NATURAL_TYPE secondary_size, NATURAL_TYPE secondary_num_slaves);
unsigned char slave_state_join(struct slave_state_vector **local_state_vectors);
unsigned char master_state_join(struct master_state_vector *result_vector, struct master_state_vector *main_vector, struct master_state_vector *secondary_vector);

// Measure probability
void slave_probability_measure_one(struct slave_state_vector **local_state_vectors);
REAL_TYPE master_probability_measure_one(struct master_state_vector *master_state, unsigned int target_index);

// Measure state
unsigned char measure_master_state_vector(struct master_state_vector *master_state, bool *measure, unsigned int target_index, struct master_state_vector *new_master_state, REAL_TYPE roll);

// Collapse state
void slave_recive_collapsed_state_vector(struct slave_state_vector **local_state_vectors, int new_state_index, NATURAL_TYPE recive_length, int slave_to_recive, NATURAL_TYPE recive_offset);
void slave_send_collapsed_state_vector(struct slave_state_vector **local_state_vectors, int old_state_index, int value, int target_index, NATURAL_TYPE send_length, int slave_to_send, NATURAL_TYPE send_offset);
void slave_send_self_state_vector(struct slave_state_vector **local_state_vectors, int old_state_index, int new_state_index, int value, int target_index, NATURAL_TYPE comm_length, NATURAL_TYPE send_offset);
void collapse_slave_state(struct slave_state_vector **local_state_vectors);
unsigned char collapse_master_state(struct master_state_vector *master_state, unsigned int target_index, int value, struct master_state_vector *new_master_state);

#endif /* QOPS_H_ */
