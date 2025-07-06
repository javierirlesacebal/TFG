#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "platform.h"
#include "qgate.h"
#include "qops.h"
#include "qstate.h"
#include "funciones_auxiliares.h"
#include <strings.h>
#include <Python.h>
#include <complex.h>
#include <errno.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <mpi.h>
#include <unistd.h>


#define ONLY_MASTER_VOID if (mpi_rank != 0) { return; }
#define ONLY_SLAVE_VOID if (mpi_rank == 0) { return; }


int mpi_rank;
int mpi_size;
MPI_Datatype MPI_DCOMPLEX;
MPI_Datatype MPI_NATURAL_TYPE;
struct qgate;


static PyObject *doki_master_registry_new(PyObject *self, PyObject *args);
static PyObject *doki_master_registry_new_data(PyObject *self, PyObject *args);
static PyObject *doki_master_registry_apply(PyObject *self, PyObject *args);
static PyObject *doki_master_registry_measure(PyObject *self, PyObject *args);
static PyObject *doki_master_registry_join(PyObject *self, PyObject *args);
static PyObject *doki_master_registry_get(PyObject *self, PyObject *args);
static PyObject *doki_master_registry_prob(PyObject *self, PyObject *args);
static PyObject *doki_master_registry_del(PyObject *self, PyObject *args);


void doki_master_registry_destroy(PyObject *master_state_capsule);
unsigned char master_send_gate(struct qgate *gate, int num_slaves);
unsigned char custom_master_state_init_py(PyObject *values, struct master_state_vector *master_state, int is_py);


void doki_registry_destroy(PyObject *capsule);
void doki_gate_destroy(PyObject *capsule);
void doki_funmatrix_destroy(PyObject *capsule);
void custom_state_init_py(PyObject *values, struct state_vector *state);
void custom_state_init_np(PyObject *values, struct state_vector *state);


static PyObject *DokiError;
static PyObject *doki_registry_new(PyObject *self, PyObject *args);
static PyObject *doki_registry_clone(PyObject *self, PyObject *args);
static PyObject *doki_registry_del(PyObject *self, PyObject *args);
static PyObject *doki_gate_new(PyObject *self, PyObject *args);
static PyObject *doki_gate_get(PyObject *self, PyObject *args);
static PyObject *doki_registry_get(PyObject *self, PyObject *args);
static PyObject *doki_registry_new_data(PyObject *self, PyObject *args);
static PyObject *doki_registry_apply(PyObject *self, PyObject *args);
static PyObject *doki_registry_join(PyObject *self, PyObject *args);
static PyObject *doki_registry_measure(PyObject *self, PyObject *args);
static PyObject *doki_registry_prob(PyObject *self, PyObject *args);
static PyObject *doki_registry_density(PyObject *self, PyObject *args);
static PyObject *doki_registry_mem(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_create(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_identity(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_densityzero(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_statezero(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_hadamard(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_addcontrol(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_get(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_add(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_sub(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_scalar_mul(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_scalar_div(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_matmul(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_ewmul(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_kron(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_eyekron(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_transpose(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_dagger(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_projection(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_shape(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_partialtrace(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_trace(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_apply(PyObject *self, PyObject *args);
static PyObject *doki_funmatrix_mem(PyObject *self, PyObject *args);


static PyMethodDef DokiMethods[] = {
	{ "master_registry_del", 		doki_master_registry_del, 		METH_VARARGS, "Destroys a master registry" },
	{ "gate_new", 					doki_gate_new, 					METH_VARARGS, "Create new gate" },
	{ "gate_get", 					doki_gate_get, 					METH_VARARGS, "Get matrix associated to gate" },
	{ "master_registry_prob", 		doki_master_registry_prob, 		METH_VARARGS, "Get the chances of obtaining 1 when measuring a certain qubit" },
	{ "master_registry_new", 		doki_master_registry_new, 		METH_VARARGS, "Create new master registry" },
	{ "master_registry_new_data", 	doki_master_registry_new_data, 	METH_VARARGS, "Create new master registry" },
	{ "master_registry_apply", 		doki_master_registry_apply, 	METH_VARARGS, "Apply a gate to a master registry" },
	{ "master_registry_measure",	doki_master_registry_measure, 	METH_VARARGS, "Measures a master registry and collapses specified qubits" },
	{ "master_registry_join", 		doki_master_registry_join, 		METH_VARARGS, "Merges two master registries" },
	{ "registry_new", 				doki_registry_new, 				METH_VARARGS, "Create new registry" },
	{ "registry_new_data", 			doki_registry_new_data, 		METH_VARARGS, "Create new registry initialized with the specified values" },
	{ "registry_clone", 			doki_registry_clone, 			METH_VARARGS, "Clone a registry" },
	{ "registry_del", 				doki_registry_del, 				METH_VARARGS, "Destroy a registry" },
	{ "registry_get", 				doki_registry_get, 				METH_VARARGS, "Get value from registry" },
	{ "master_registry_get", 		doki_master_registry_get, 		METH_VARARGS, "Get value from master registry" },
	{ "registry_apply", 			doki_registry_apply, 			METH_VARARGS, "Apply a gate" },
	{ "registry_join", 				doki_registry_join, 			METH_VARARGS, "Merges two registries" },
	{ "registry_measure", 			doki_registry_measure, 			METH_VARARGS, "Measures and collapses specified qubits" },
	{ "registry_prob", 				doki_registry_prob, 			METH_VARARGS, "Get the chances of obtaining 1 when measuring a certain qubit" },
	{ "registry_density", 			doki_registry_density, 			METH_VARARGS, "Get the density matrix" },
	{ "registry_mem", 				doki_registry_mem, 				METH_VARARGS, "Get the memory allocated by this registry in bytes" },
	{ "funmatrix_create", 			doki_funmatrix_create, 			METH_VARARGS, "Create a functional matrix from a matrix" },
	{ "funmatrix_identity", 		doki_funmatrix_identity, 		METH_VARARGS, "Create an identity functional matrix of the specified number of qubits" },
	{ "funmatrix_statezero", 		doki_funmatrix_statezero, 		METH_VARARGS, "Create a functional matrix representing the state vector of a quantum system of n qubits at state zero" },
	{ "funmatrix_densityzero", 		doki_funmatrix_densityzero, 	METH_VARARGS, "Create a functional matrix representing the density matrix of a quantum system of n qubits at state zero" },
	{ "funmatrix_hadamard", 		doki_funmatrix_hadamard, 		METH_VARARGS, "Create a functional matrix from a Hadamard gate of the specified number of qubits" },
	{ "funmatrix_addcontrol", 		doki_funmatrix_addcontrol, 		METH_VARARGS, "Takes a gate as an input and returns the same gate with a control qubit" },
	{ "funmatrix_get", 				doki_funmatrix_get, 			METH_VARARGS, "Get a value from a functional matrix" },
	{ "funmatrix_add", 				doki_funmatrix_add, 			METH_VARARGS, "Get the addition of two functional matrices" },
	{ "funmatrix_sub", 				doki_funmatrix_sub, 			METH_VARARGS, "Get the substraction of two functional matrices" },
	{ "funmatrix_scalar_mul", 		doki_funmatrix_scalar_mul, 		METH_VARARGS, "Get the product of a scalar and a functional matrix" },
	{ "funmatrix_scalar_div", 		doki_funmatrix_scalar_div, 		METH_VARARGS, "Get the division of a functional matrix by a scalar" },
	{ "funmatrix_matmul", 			doki_funmatrix_matmul, 			METH_VARARGS, "Get the matrix product of two functional matrices" },
	{ "funmatrix_ewmul", 			doki_funmatrix_ewmul, 			METH_VARARGS, "Get the entity-wise multiplication of two functional matrices" },
	{ "funmatrix_kron", 			doki_funmatrix_kron, 			METH_VARARGS, "Get the Kronecker product of two functional matrices" },
	{ "funmatrix_eyekron", 			doki_funmatrix_eyekron, 		METH_VARARGS, "Get the Kronecker product of I(2^left), U and I(2^right)" },
	{ "funmatrix_transpose", 		doki_funmatrix_transpose, 		METH_VARARGS, "Get the transpose of a functional matrix" },
	{ "funmatrix_dagger", 			doki_funmatrix_dagger, 			METH_VARARGS, "Get the conjugate-transpose of a functional matrix" },
	{ "funmatrix_projection", 		doki_funmatrix_projection, 		METH_VARARGS, "Get the result of a projection over a column vector" },
	{ "funmatrix_shape", 			doki_funmatrix_shape, 			METH_VARARGS, "Get a tuple with the shape of the matrix" },
	{ "funmatrix_partialtrace", 	doki_funmatrix_partialtrace, 	METH_VARARGS, "Get the partial trace of a functional matrix" },
	{ "funmatrix_trace", 			doki_funmatrix_trace, 			METH_VARARGS, "Get the trace of a functional matrix" },
	{ "funmatrix_apply", 			doki_funmatrix_apply, 			METH_VARARGS, "Get the resulting functional matrix after applying a gate to a state vector" },
	{ "funmatrix_mem", 				doki_funmatrix_mem, 			METH_VARARGS, "Get the memory allocated by this FMatrix in bytes" },
	{ NULL, NULL, 0, NULL } /* Sentinel */
};

static struct PyModuleDef dokimodule = { PyModuleDef_HEAD_INIT, "doki", NULL, -1, DokiMethods };

// Revisada
static void initialize_mpi() {
    int flag;
    MPI_Initialized(&flag);

    if (!flag) {
        int argc = 0;
        char** argv = NULL;
        MPI_Init(&argc, &argv);
    }
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_DCOMPLEX);
	MPI_Type_commit(&MPI_DCOMPLEX);
	MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(NATURAL_TYPE), &MPI_NATURAL_TYPE);
	printf("Rank %d inicializa el modulo\n", mpi_rank);
}

// Revisada
static void finalize_mpi() {
	int flag, i;
	char dummy;
	dummy = 42;
	printf("Rank %d Finalizando MPI...\n", mpi_rank);
	fflush(stdout);
    MPI_Finalized(&flag);
    if (!flag) {
		if (mpi_rank == 0) {
			for (i = 1; i < mpi_size; i++) {
				send_ints(i, 2, 0, 0);
			}
			for (int i = 1; i < mpi_size; i++) {
				MPI_Recv(&dummy, 1, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				fflush(stdout);
			}
		} else {
			MPI_Send(&dummy, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
		}
		MPI_Finalize();
		exit(0);
    }
}

// Revisada
static PyObject *doki_master_registry_prob(PyObject *self, PyObject *args) {
	PyObject *capsule;
	void *raw_master_state;
	struct master_state_vector *master_state;
	unsigned int id;

	if (!PyArg_ParseTuple(args, "OI", &capsule, &id)) {
		PyErr_SetString(DokiError, "Syntax: registry_prob(registry, qubit_id)");
		return NULL;
	}

	raw_master_state = PyCapsule_GetPointer(capsule, "qsimov.doki.master_state_vector");
	if (raw_master_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to master registry");
		return NULL;
	}
	master_state = (struct master_state_vector *)raw_master_state;
	return PyFloat_FromDouble(master_probability_measure_one(master_state, id));
}

// Revisada
static void clear_slave_qgate(struct qgate *gate){
	int i;
    if (gate->matrix != NULL) {
        for (int i = 0; i < gate->size; i++) {
			SAFE_FREE(gate->matrix[i]);
        }
		SAFE_FREE(gate->matrix);
    }
}

// Revisada
static void command_slave_free_state(struct slave_state_vector **local_state_vectors, int index){
	if (local_state_vectors[index] != NULL) {
		SAFE_FREE(local_state_vectors[index]->vector);
		SAFE_FREE(local_state_vectors[index]);
	}
}

// VERBOSED
unsigned char command_slave_state_init(struct slave_state_vector **local_state_vectors){
	int i, index;
	int request[3];
	unsigned char result;
	
	MPI_Recv(&request, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	// Comprobar cual es el primer hueco en "local_state_vectors"
	index = -1;
	for (i = 0; i < SLAVE_MAX_SV; i++) {
		if (local_state_vectors[i] == NULL){
			index = i;
			break;
		}
	}
	if (index == -1){
		send_ints(0, 2, 0, 0);
		return 1;
	}

	// Instanciacion del estado
	local_state_vectors[index] = MALLOC_TYPE(1, struct slave_state_vector);
	if (local_state_vectors[index] == NULL) {
		send_ints(0, 2, 0, 0);
		return 2;
	}
		
	// Inicializacion del estado
	result = slave_state_init(local_state_vectors[index], request[0], request[1], request[2], mpi_rank-1);
	if (result == 1) {
		free(local_state_vectors[index]);
		local_state_vectors[index] = NULL;
		send_ints(0, 2, 0, 0);
		return 3;
	}
	
	// Envía mensaje de finalizacion correcta y retorna 0
	send_ints(0, 2, 1, index);
	return 0;
}

// Revisada
void command_slave_recive_qgate(struct qgate *gate, int num_qubits){
	COMPLEX_TYPE value;
	int errored;
	int i, j;
	int response;
	// Genera una nueva puerta
	errored = 0;
	clear_slave_qgate(gate);
	
	// Establece el numero de num_qubits y el size
	gate->num_qubits = num_qubits;
	gate->size = NATURAL_ONE << num_qubits;

	// Reserva toda la matriz de la puerta
	SAFE_MALLOC(gate->matrix, gate->size, COMPLEX_TYPE *, errored, 1, cleanup);
	for (i = 0; i < gate->size; i++) {
		SAFE_MALLOC(gate->matrix[i], gate->size, COMPLEX_TYPE, errored, 1, cleanup);
	}
	
	// Envia confirmacion de que está listo para recibir datos y espera otra
	MPI_NOTIFY_READY(response, cleanup);
	
	// Recive la informacion de la puerta 
	for (i = 0; i < gate->size; i++) {
		MPI_Recv(gate->matrix[i], gate->size, MPI_DCOMPLEX, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
	}

cleanup:
	if (errored == 1){
		MPI_SEND_ABORT(response);
	}
}


// Revisada
static void slave_loop() {
	ONLY_SLAVE_VOID
	struct slave_state_vector *local_state_vectors[SLAVE_MAX_SV] = {NULL};
	struct qgate gate;
	int msg[2];
	int command;
	gate.matrix = NULL;
	
	while (1) {
		// Recibe el comando
		MPI_Recv(&msg, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		// Comprueba si el mensaje es el de terminacion
		command = msg[0];
		if (command == 0) {
			break;
		}
		
		// En funcion del comando recibido, ejecuta la accion correspondiente
		switch (command) {
			case 1:
				command_slave_state_init(local_state_vectors);
				break;
			case 2:
				command_slave_free_state(local_state_vectors, msg[1]);
				break;
			case 3:
				command_slave_recive_qgate(&gate, msg[1]);
				break;
			case 4:
				apply_gate_to_slave_state(local_state_vectors, &gate);
				break;
			case 5: 
				slave_probability_measure_one(local_state_vectors);
				break;
			case 6: 
				slave_state_clone(local_state_vectors);
				break;
			case 7: 
				collapse_slave_state(local_state_vectors);
				break;
			case 8: 
				print_slave_state(local_state_vectors, msg[1]);
				break;
			case 9:
				slave_recive_state_vector(local_state_vectors);
				break;
			case 10:
				slave_state_join(local_state_vectors);
				break;
			case 11:
				slave_state_normalize(local_state_vectors, msg[1]);
				break;
			case 12:
				slave_send_vector_element(local_state_vectors, msg[1]);
				break;
			default:
				break;
		}
	}
	clear_slave_qgate(&gate);
	finalize_mpi();
	exit(0);
}

// Revisada
PyMODINIT_FUNC PyInit_doki(void) {
	PyObject *m;
	assert(!PyErr_Occurred());
	import_array();
	
	// Registrar MPI_Finalize con atexit
    Py_AtExit(finalize_mpi);
	m = PyModule_Create(&dokimodule);
	if (m == NULL){
		return NULL;
	}
	DokiError = PyErr_NewException("qsimov.doki.error", NULL, NULL);
	Py_XINCREF(DokiError);
	if (PyModule_AddObject(m, "error", DokiError) < 0) {
		Py_XDECREF(DokiError);
		Py_CLEAR(DokiError);
		Py_DECREF(m);
		return NULL;
	}
	
	// Inicializar MPI
	initialize_mpi();
	slave_loop();
	return m;
}

// Revisada
int main(int argc, char *argv[]) {
	wchar_t *program = Py_DecodeLocale(argv[0], NULL);
	if (program == NULL) {
		fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
		exit(1);
	}

	// Add a built-in module, before Py_Initialize
	if (PyImport_AppendInittab("doki", PyInit_doki) == -1) {
		fprintf(stderr, "Error: could not extend in-built modules table\n");
		exit(1);
	}

	// Initialize the Python interpreter
	Py_Initialize();

	// Import the module
	PyObject *pmodule = PyImport_ImportModule("doki");
	if (!pmodule) {
		PyErr_Print();
		fprintf(stderr, "Error: could not import module 'doki'\n");
	}
	
	PyMem_RawFree(program);
	return 0;
}

// Revisada
static PyObject *doki_master_registry_new(PyObject *self, PyObject *args) {
	int num_qubits, debug_enabled;
	unsigned char result;
	struct master_state_vector *master_state;
	
	if (!PyArg_ParseTuple(args, "ii", &num_qubits, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_new(num_qubits, verbose)");
		return NULL;
	}
	if (num_qubits == 0) {
		PyErr_SetString(DokiError, "num_qubits can't be zero");
		return NULL;
	}
	
	master_state = MALLOC_TYPE(1, struct master_state_vector);
	if (master_state == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate master_state structure");
		return NULL;
	}
	
	result = master_state_init(master_state, num_qubits, 1, (mpi_size-1));
	if (result == 1) {
		PyErr_SetString(DokiError, "Number of qubits exceeds maximum of master_state_vector\n");
		return NULL;
	} else if (result == 2) {
		PyErr_SetString(DokiError, "Number of slaves must be a power of 2\n");
		return NULL;
	} else if (result == 3) {
		PyErr_SetString(DokiError, "Failed allocating slave_ids of master_state_vector\n");
		return NULL;
	} else if (result == 4) {
		PyErr_SetString(DokiError, "One slave failed to allocate slave_state_vector\n");
		return NULL;
	} else if (result != 0) {
		PyErr_SetString(DokiError, "Unknown error when creating master_state_vector\n");
		return NULL;
	}

	return PyCapsule_New((void *)master_state, "qsimov.doki.master_state_vector", doki_master_registry_destroy);
}

// Revisada
static PyObject *doki_master_registry_new_data(PyObject *self, PyObject *args) {
	PyObject *raw_vals;
	int num_qubits;
	unsigned char result;
	struct master_state_vector *master_state;
	short debug_enabled;


	if (!PyArg_ParseTuple(args, "iOh", &num_qubits, &raw_vals, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_new_data(num_qubits, values, verbose)");
		return NULL;
	}
	if (num_qubits == 0) {
		PyErr_SetString(DokiError, "num_qubits can't be zero");
		return NULL;
	}
	master_state = MALLOC_TYPE(1, struct master_state_vector);
	if (master_state == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate master_state structure");
		return NULL;
	}
	

	result = master_state_init(master_state, num_qubits, 1, (mpi_size-1) );
	if (result != 0) {
		PyErr_SetString(DokiError, "ERROR IN master_state_init()");
		return NULL;
	}

	if (PyArray_Check(raw_vals)) {
		PyArrayObject *array = (PyArrayObject *)raw_vals;
		if (!PyArray_ISNUMBER(array)) {
			PyErr_SetString(DokiError, "values have to be numbers");
			return NULL;
		}
		if (PyArray_SIZE(array) != master_state->size) {
			PyErr_SetString(DokiError, "Wrong array size for the specified number of qubits");
			return NULL;
		}
		custom_master_state_init_py(raw_vals, master_state, 0);
	} else if (PyList_Check(raw_vals)) {
		if (PyList_GET_SIZE(raw_vals) != master_state->size) {
			PyErr_SetString(DokiError, "Wrong list size for the specified number of qubits\n");
			return NULL;
		}
		custom_master_state_init_py(raw_vals, master_state, 1);
	} else {
		PyErr_SetString(DokiError, "values has to be either a python list or a numpy array");
		return NULL;
	}
	
	return PyCapsule_New((void *)master_state, "qsimov.doki.master_state_vector", doki_master_registry_destroy);
}

// Revisada
PyObject *aux_getter(PyObject *values, NATURAL_TYPE i, int is_py){
	PyObject *aux;
	if(is_py){
		aux = PyList_GetItem(values, i);
	} else {
		PyArrayObject *array = (PyArrayObject *)values;
		aux = PyArray_GETITEM(array, PyArray_GETPTR1(array, i));
	}
	return aux;
}

// Revisada
unsigned char custom_master_state_init_py(PyObject *values, struct master_state_vector *master_state, int is_py) {
	NATURAL_TYPE i, slave_elements, chunk_index;
	unsigned char errored;
	int parameters[3];
	PyObject *aux;
	COMPLEX_TYPE *chunk;
	int chunk_size, chunks_per_slave;
	int lsb_bitsize, esclavo;
	int msg[2];
	
	errored = 0;
	slave_elements = master_state->size / master_state->num_slaves;
	chunk_size = (slave_elements > SLAVE_COMMUNICATION_CHUNK_SIZE) ? SLAVE_COMMUNICATION_CHUNK_SIZE : slave_elements;
	chunks_per_slave = slave_elements / chunk_size;
	
	SAFE_MALLOC(chunk, chunk_size, COMPLEX_TYPE, errored, 1, cleanup);
	send_slaves_ints(master_state, 2, 9, chunks_per_slave);
	
	for (i = 0; i < master_state->num_slaves; i++) {
		send_ints(i+1, 3, master_state->slave_ids[i], chunks_per_slave, chunk_size);
	}
	
	lsb_bitsize = master_state->num_qubits - (ffs(master_state->num_slaves) - 1);
	chunk_index = 0;
	for (i = 0; i < master_state->size; i++) {
		aux = aux_getter(values, i, is_py);
		chunk[chunk_index] = COMPLEX_INIT(PyComplex_RealAsDouble(aux), PyComplex_ImagAsDouble(aux));
		chunk_index++;
		if (chunk_index >= chunk_size){	
			esclavo = (i >> lsb_bitsize) + 1;
			MPI_Send(chunk, chunk_size, MPI_DOUBLE_COMPLEX, esclavo, 0, MPI_COMM_WORLD);
			chunk_index = 0;
		}
	}
	
cleanup:
	return errored;
}

// Revisada
void doki_master_registry_destroy(PyObject *master_state_capsule) {
	struct master_state_vector *master_state;
	void *raw_state;
	raw_state = PyCapsule_GetPointer(master_state_capsule, "qsimov.doki.master_state_vector");

	if (raw_state != NULL) {
		master_state = (struct master_state_vector *)raw_state;
		master_state_free(master_state);
	}
}

// Revisada
static PyObject *doki_master_registry_apply(PyObject *self, PyObject *args) {
	PyObject *raw_val, *master_state_capsule, *gate_capsule, *target_list, *control_set, *acontrol_set, *aux;
	void *raw_state = NULL, *raw_gate = NULL;
	struct master_state_vector *master_state, *new_master_state;
	struct qgate *gate;
	unsigned char exit_code;
	unsigned int num_targets, num_controls, num_anticontrols, i;
	unsigned int *targets = NULL, *controls = NULL, *anticontrols = NULL;
	int debug_enabled;

	// Parsear los argumentos
	if (!PyArg_ParseTuple(args, "OOOOOp", &master_state_capsule, &gate_capsule, &target_list, &control_set, &acontrol_set, &debug_enabled)) {
		PyErr_SetString( DokiError, "Syntax: registry_apply(registry, gate, target_list, control_set, anticontrol_set, verbose)");
		goto cleanup;
	}
	
	//  Comprobar que es un puntero basico y castearlo a master_state_vector
	raw_state = PyCapsule_GetPointer(master_state_capsule, "qsimov.doki.master_state_vector");
	if (raw_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to master registry");
		goto cleanup;
	}
	master_state = (struct master_state_vector *)raw_state;
	
	//  Comprobar que es un puntero basico y castearlo a master_state_vector
	raw_gate = PyCapsule_GetPointer(gate_capsule, "qsimov.doki.gate");
	if (raw_gate == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to gate");
		goto cleanup;
	}
	gate = (struct qgate *)raw_gate;
	
	// Comprobar que "target_list" es de tipo lista
	if (!PyList_Check(target_list)) {
		PyErr_SetString(DokiError, "target_list must be a list");
		goto cleanup;
	}
	
	// Comprueba que el numero de "targets" sea del mismo tamaño que el numero de qubits de la puerta
	num_targets = (unsigned int)PyList_Size(target_list);
	if (num_targets != gate->num_qubits) {
		PyErr_SetString( DokiError, "Wrong number of targets specified for that gate");
		goto cleanup;
	}
	
	// Comprueba que el numero de bits de control sean 0 o Py_None
	if (control_set == Py_None) {
		num_controls = 0;
	} else if (PySet_Check(control_set)) {
		num_controls = (unsigned int)PySet_Size(control_set);
	} else {
		PyErr_SetString(DokiError, "control_set must be a set or None");
		goto cleanup;
	}
	
	// Comprueba que el numero de bits de anti-control sean 0 o Py_None
	if (acontrol_set == Py_None) {
		num_anticontrols = 0;
	} else if (PySet_Check(acontrol_set)) {
		num_anticontrols = (unsigned int)PySet_Size(acontrol_set);
	} else {
		PyErr_SetString(DokiError, "anticontrol_set must be a set or None");
		goto cleanup;
	}
	
	// Aloca memoria para guardar los "targets" qubits
	targets = MALLOC_TYPE(num_targets, unsigned int);
	if (targets == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate target array");
		goto cleanup;
	}
	
	// En caso de que haya bits de control, alloca un vector que los contenga
	if (num_controls > 0) {
		controls = MALLOC_TYPE(num_controls, unsigned int);
		if (controls == NULL) {
			PyErr_SetString(DokiError, "Failed to allocate control array");
			goto cleanup;
		}
	}
	
	// En caso de que haya bits de anti_control, alloca un vector que los contenga
	if (num_anticontrols > 0) {
		anticontrols = MALLOC_TYPE(num_anticontrols, unsigned int);
		if (anticontrols == NULL) {
			PyErr_SetString(DokiError, "Failed to allocate anticontrol array");
			goto cleanup;
		}
	}
	
	// Convierte la lista de python a un array de C valido con los "control_set"
	if (num_controls > 0) {
		aux = PySet_New(control_set);
		for (i = 0; i < num_controls; i++) {
			raw_val = PySet_Pop(aux);
			// Por cada elemento de "control_set" lo extrae y comprueba que sea de tipo long
			if (!PyLong_Check(raw_val)) {
				PyErr_SetString( DokiError, "control_set must be a set qubit ids (unsigned integers)");
				goto cleanup;
			}
			// Almacena el valor y comprueba que este dentro del rango de qubits del estado
			controls[i] = PyLong_AsLong(raw_val);
			if (controls[i] >= master_state->num_qubits) {
				PyErr_SetString(DokiError, "Control qubit out of range");
				goto cleanup;;
			}
		}
	}
	
	// Convierte la lista de python a un array de C valido con los "acontrol_set"
	if (num_anticontrols > 0) {
		aux = PySet_New(acontrol_set);
		for (i = 0; i < num_anticontrols; i++) {
			raw_val = PySet_Pop(aux);
			// Por cada elemento de "acontrol_set" lo extrae y comprueba que sea de tipo long
			if (!PyLong_Check(raw_val)) {
				PyErr_SetString(DokiError, "anticontrol_set must be a set qubit ids (unsigned integers)");
				goto cleanup;
			}
			// Comprueba que no este en el "control_set"
			if (PySet_Contains(control_set, raw_val)) {
				PyErr_SetString(DokiError, "A control cannot also be an anticontrol");
				goto cleanup;
			}
			// Almacena el valor y comprueba que este dentro del rango de qubits del estado
			anticontrols[i] = PyLong_AsLong(raw_val);
			if (anticontrols[i] >= master_state->num_qubits) {
				PyErr_SetString( DokiError, "Anticontrol qubit out of range");
				goto cleanup;
			}
		}
	}
	
	
	// Por cada qbit objetivo, extrae su indice, comprobando que es un long (entero)
	for (i = 0; i < num_targets; i++) {
		raw_val = PyList_GetItem(target_list, i);
		if (!PyLong_Check(raw_val)) {
			PyErr_SetString(DokiError, "target_list must be a list of qubit ids (unsigned integers)");
			goto cleanup;
		}
		// Comprueba que el qbit objetivo no sea qbit de control o anti-control
		if ((num_controls > 0 && PySet_Contains(control_set, raw_val)) || (num_anticontrols > 0 && PySet_Contains(acontrol_set, raw_val))) {
			PyErr_SetString(DokiError, "A target cannot also be a control or an anticontrol");
			goto cleanup;
		}
		// Guarda el indice del qbit objetivo y comprueba que el indice no se salga del rango del vector de estado
		targets[i] = PyLong_AsLong(raw_val);
		if (targets[i] >= master_state->num_qubits) {
			PyErr_SetString(DokiError, "Target qubit out of range");
			goto cleanup;
		}
	}
	
	// Aloca un nuevo "master_state_vector" que contendrá el resultado de la operacion
	new_master_state = MALLOC_TYPE(1, struct master_state_vector);
	if (new_master_state == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate new master_state structure");
		goto cleanup;
	}
	
	
	// Envía la puerta a los esclavos
	exit_code = master_send_gate(gate, master_state->num_slaves);
	
	switch (exit_code) {
    case 1:
        PyErr_SetString(DokiError, "gate can not be NULL");
        break;
    case 2:
        PyErr_SetString(DokiError, "Slaves failed to get the gate");
        break;
    default:
        if (exit_code != 0) {
            PyErr_SetString(DokiError, "Unknown error when transfering gate");
        }
        break;
	}
	if (exit_code) {
		PyErr_SetString(DokiError, "Gate transfer to slaves failed");
		return NULL;
	}
	
	exit_code = apply_gate_to_master_state(master_state, new_master_state, num_targets, targets, num_controls,  controls, num_anticontrols, anticontrols);
	
	
	switch (exit_code) {
    case 1:
        PyErr_SetString(DokiError, "num_qubits es mayor que MAX_NUM_QUBITS");
        break;
    case 2:
        PyErr_SetString(DokiError, "Numero de esclavos debe de ser potencia de 2");
        break;
    case 3:
        PyErr_SetString(DokiError, "Fallo al alocar 'slave_ids'");
        break;
    case 4:
        PyErr_SetString(DokiError, "Fallo al crear los 'slave_state_vector'");
        break;
    case 5:
        PyErr_SetString(DokiError, "'new_master_state' es NULL");
        break;
    case 11:
        PyErr_SetString(DokiError, "Failed to allocate not_copy structure");
        break;
    default:
        if (exit_code != 0) {
            PyErr_SetString(DokiError, "Unknown error when applying gate");
        }
        break;
}

cleanup:
	// Libera la memoria
	SAFE_FREE(targets);
	SAFE_FREE(controls);
	SAFE_FREE(anticontrols);
	
	// Si falla retorna NULL
	if (exit_code != 0){
		return NULL;
	}
	// Retorna el resultado
	return PyCapsule_New((void *)new_master_state, "qsimov.doki.master_state_vector", doki_master_registry_destroy);
}


static PyObject *doki_master_registry_measure(PyObject *self, PyObject *args) {
	PyObject *capsule, *py_measured_val, *result, *new_capsule, *roll_list;
	Py_ssize_t roll_id;
	void *raw_master_state;
	struct master_state_vector *master_state, *new_master_state, *aux_master_state;
	NATURAL_TYPE mask;
	REAL_TYPE roll;
	unsigned int i, curr_id, initial_num_qubits, measured_qty;
	_Bool measure_id, measured_val;
	unsigned char exit_code;
	int debug_enabled;
	
	if (!PyArg_ParseTuple(args, "OKOp", &capsule, &mask, &roll_list, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_measure(registry, mask, roll_list, verbose)");
		return NULL;
	}
	
	raw_master_state = PyCapsule_GetPointer(capsule, "qsimov.doki.master_state_vector");
	if (raw_master_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry");
		return NULL;
	}
	
	if (!PyList_Check(roll_list)) {
		PyErr_SetString(DokiError, "Roll_list must be a list of real numbers in [0, 1)!");
		return NULL;
	}
	
	master_state = (struct master_state_vector *) raw_master_state;
	initial_num_qubits = master_state->num_qubits;
	result = PyList_New(initial_num_qubits);

	new_master_state = MALLOC_TYPE(1, struct master_state_vector);
	if (new_master_state == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate new master master_state structure");
		return NULL;
	}
		
	exit_code = master_state_clone(new_master_state, master_state);
	
	if (exit_code == 1) {
		PyErr_SetString(DokiError, "Failed to allocate master_state vector");
		return NULL;
	} else if (exit_code == 2) {
		PyErr_SetString(DokiError, "Failed to allocate master_state chunk");
		return NULL;
	} else if (exit_code == 3) {
		PyErr_SetString(DokiError, "Wrong number of qubits");
		return NULL;
	} else if (exit_code != 0) {
		PyErr_SetString(DokiError, "Unknown error when cloning master_state");
		return NULL;
	}

	measured_qty = 0;
	roll_id = 0;
	aux_master_state = NULL;
	for (i = 0; i < initial_num_qubits; i++) {
		curr_id = initial_num_qubits - i - 1;
		measure_id = mask & (NATURAL_ONE << curr_id);
		py_measured_val = Py_None;
		if (measure_id) {
			if (new_master_state == NULL || new_master_state->num_qubits == 0) {
				master_state_free(new_master_state);
				PyErr_SetString(DokiError, "Could not measure non_existant qubits");
				return NULL;
			}
			// Obtiene de la lista de roll el valor y comprueba que esté entre 0 y 1
			roll = PyFloat_AsDouble( PyList_GetItem(roll_list, roll_id));
			if (roll < 0 || roll >= 1) {
				master_state_free(new_master_state);
				PyErr_SetString(DokiError, "roll not in interval [0, 1)!");
				return NULL;
			}
			roll_id++;
			
			// Alloca un nuevo vector de estado
			aux_master_state = MALLOC_TYPE(1, struct master_state_vector);
			if (aux_master_state == NULL) {
				master_state_free(new_master_state);
				PyErr_SetString(DokiError, "Failed to allocate aux_master_state master_state structure");
				return NULL;
			}
			
			// Realiza la medicion del valor
			exit_code = measure_master_state_vector(new_master_state, &measured_val, curr_id, aux_master_state, roll);
			if (exit_code != 0) {
				master_state_free(aux_master_state);
				break;
			}
			
			// Si la constante de normalizacion es 0 termina con un error
			if (aux_master_state->num_qubits > 0 && aux_master_state->norm_const == 0.0) {
				master_state_free(aux_master_state);
				master_state_free(new_master_state);
				PyErr_SetString(DokiError, "New normalization constant is 0. Please report this error with the steps to reproduce it.");
				return NULL;
			}
			measured_qty++;
			py_measured_val = measured_val ? Py_True : Py_False;
			master_state_free(new_master_state);
			new_master_state = aux_master_state;
			aux_master_state = NULL;
		}
		PyList_SET_ITEM(result, i, py_measured_val);
	}
	
	if (exit_code != 0) {
		master_state_free(new_master_state);
		return NULL;
	}

	if (master_state->num_qubits - measured_qty > 0) {
		new_capsule = PyCapsule_New((void *)new_master_state, "qsimov.doki.master_state_vector", doki_master_registry_destroy);
	} else {
		master_state_free(new_master_state);
		new_capsule = Py_None;
	}

	return PyTuple_Pack(2, new_capsule, result);
}


unsigned char master_send_gate(struct qgate *gate, int num_slaves){
	int i, j, errored;
	int recive_gate_request[2];
	int response[1];
	
	if (gate == NULL) {
		return 1;
	}
	
	// Poner los esclavos en modo rececpcion de puerta
	for (i = 0; i < num_slaves; i++){
		send_ints(i+1, 2, 3, gate->num_qubits);
	}
	
	// Espera las confirmaciones
	errored = 0;
	for (i = 0; i < num_slaves; i++){
		MPI_Recv(&response, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (!response[0]){
			errored = 1;
		}
	}
	// Envia las respuestas a todos los esclavos
	response[0] = !errored;
	for (i = 0; i < num_slaves; i++){
		MPI_Send(&response, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD);
	}
	
	// Si algun esclavo ha fallado, indica al resto que aborten la rececpcion
	if (errored){
		return 2;
	}

	// Enviar la informacion
	for (j = 0; j < gate->size; j++){
		for (i = 0; i < num_slaves; i++){
			MPI_Send(gate->matrix[j], gate->size, MPI_DCOMPLEX, i+1, 0, MPI_COMM_WORLD);
		}
	}
	
	return 0;
}


static PyObject *doki_master_registry_join(PyObject *self, PyObject *args) {
	PyObject *capsule1, *capsule2;
	void *raw_master_state1, *raw_master_state2;
	struct master_state_vector *master_state1, *master_state2, *result;
	unsigned char exit_code;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OOp", &capsule1, &capsule2, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: master_registry_join(most_registry, least_registry, verbose)");
		return NULL;
	}

	raw_master_state1 = PyCapsule_GetPointer(capsule1, "qsimov.doki.master_state_vector");
	if (raw_master_state1 == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry1");
		return NULL;
	}

	raw_master_state2 = PyCapsule_GetPointer(capsule2, "qsimov.doki.master_state_vector");
	if (raw_master_state2 == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry2");
		return NULL;
	}
	
	master_state1 = (struct master_state_vector *)raw_master_state1;
	master_state2 = (struct master_state_vector *)raw_master_state2;
	result = MALLOC_TYPE(1, struct master_state_vector);
	if (result == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate new state structure");
		return NULL;
	}
	
	exit_code = master_state_join(result, master_state1, master_state2);
	
	if (exit_code != 0) {
		switch (exit_code) {
		case 1:
			PyErr_SetString(DokiError, "ERROR 1 joining master states");
			break;
		default:
			PyErr_SetString(DokiError, "Unknown error when joining master states");
		}
		return NULL;
	}

	return PyCapsule_New((void *)result, "qsimov.doki.master_state_vector", doki_master_registry_destroy);
}

// Revisada
void doki_registry_destroy(PyObject *capsule) {
	struct state_vector *state;
	void *raw_state;
	raw_state = PyCapsule_GetPointer(capsule, "qsimov.doki.state_vector");

	if (raw_state != NULL) {
		state = (struct state_vector *)raw_state;
		state_clear(state);
		free(state);
	}
}

// Revisada
void doki_gate_destroy(PyObject *capsule) {
	struct qgate *gate;
	void *raw_gate;
	NATURAL_TYPE i;

	raw_gate = PyCapsule_GetPointer(capsule, "qsimov.doki.gate");

	if (raw_gate != NULL) {
		gate = (struct qgate *)raw_gate;
		for (i = 0; i < gate->size; i++) {
			free(gate->matrix[i]);
		}
		free(gate->matrix);
		free(gate);
	}
}


void doki_funmatrix_destroy(PyObject *capsule) {
	struct FMatrix *matrix;
	void *raw_matrix;

	raw_matrix = PyCapsule_GetPointer(capsule, "qsimov.doki.funmatrix");
	if (raw_matrix != NULL) {
		matrix = (struct FMatrix *)raw_matrix;
		FM_destroy(matrix);
	}
}

static PyObject *doki_registry_new(PyObject *self, PyObject *args) {
	unsigned int num_qubits;
	unsigned char result;
	struct state_vector *state;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Ip", &num_qubits, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_new(num_qubits, verbose)");
		return NULL;
	}
	if (num_qubits == 0) {
		PyErr_SetString(DokiError, "num_qubits can't be zero");
		return NULL;
	}

	state = MALLOC_TYPE(1, struct state_vector);
	if (state == NULL) {
		PyErr_SetString(DokiError,
				"Failed to allocate state structure");
		return NULL;
	}
	result = state_init(state, num_qubits, true);
	if (result == 1) {
		PyErr_SetString(DokiError, "Failed to allocate state vector");
		return NULL;
	} else if (result == 2) {
		PyErr_SetString(DokiError, "Failed to allocate state chunk");
		return NULL;
	} else if (result == 3) {
		PyErr_SetString(DokiError, "Number of qubits exceeds maximum");
		return NULL;
	} else if (result != 0) {
		PyErr_SetString(DokiError, "Unknown error when creating state");
		return NULL;
	}
	return PyCapsule_New((void *)state, "qsimov.doki.state_vector", &doki_registry_destroy);
}

static PyObject *doki_registry_clone(PyObject *self, PyObject *args){
	PyObject *source_capsule;
	unsigned char result;
	void *raw_source;
	struct state_vector *source, *dest;
	int num_threads, debug_enabled;

	if (!PyArg_ParseTuple(args, "Oip", &source_capsule, &num_threads,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: registry_clone(registry, num_threads, verbose)");
		return NULL;
	}

	if (num_threads <= 0 && num_threads != -1) {
		PyErr_SetString(
			DokiError,
			"num_threads must be at least 1 (or -1 to let OpenMP choose)");
		return NULL;
	}

	raw_source = PyCapsule_GetPointer(source_capsule,
					  "qsimov.doki.state_vector");
	if (raw_source == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to source registry");
		return NULL;
	}
	source = (struct state_vector *)raw_source;

	dest = MALLOC_TYPE(1, struct state_vector);
	if (dest == NULL) {
		PyErr_SetString(DokiError,
				"Failed to allocate new state structure");
		return NULL;
	}

	if (num_threads != -1) {
		omp_set_num_threads(num_threads);
	}

	result = state_clone(dest, source);
	if (result == 1) {
		PyErr_SetString(DokiError, "Failed to allocate state vector");
		return NULL;
	} else if (result == 2) {
		PyErr_SetString(DokiError, "Failed to allocate state chunk");
		return NULL;
	} else if (result != 0) {
		PyErr_SetString(DokiError, "Unknown error when cloning state");
		return NULL;
	}
	return PyCapsule_New((void *)dest, "qsimov.doki.state_vector", &doki_registry_destroy);
}

static PyObject *doki_master_registry_del(PyObject *self, PyObject *args) {
	PyObject *capsule;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "O", &capsule)) {
		PyErr_SetString(DokiError, "Syntax: master_registry_del(registry)");
		return NULL;
	}

	doki_master_registry_destroy(capsule);
	PyCapsule_SetDestructor(capsule, NULL);

	Py_RETURN_NONE;
}

// Revisada
static PyObject *doki_registry_del(PyObject *self, PyObject *args) {
	PyObject *capsule;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &capsule, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_del(registry, verbose)");
		return NULL;
	}

	doki_registry_destroy(capsule);
	PyCapsule_SetDestructor(capsule, NULL);

	Py_RETURN_NONE;
}

static PyObject *doki_gate_new(PyObject *self, PyObject *args) {
	PyObject *list, *row, *raw_val;
	unsigned int num_qubits;
	NATURAL_TYPE i, j, k;
	COMPLEX_TYPE val;
	struct qgate *gate;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "IOp", &num_qubits, &list, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: gate_new(num_qubits, gate, verbose)");
		return NULL;
	}
	if (num_qubits == 0) {
		PyErr_SetString(DokiError, "num_qubits can't be zero");
		return NULL;
	}
	if (!PyList_Check(list)) {
		PyErr_SetString(DokiError,
				"gate must be a list of lists (matrix)");
		return NULL;
	}

	gate = MALLOC_TYPE(1, struct qgate);
	if (gate == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate qgate");
		return NULL;
	}

	gate->num_qubits = num_qubits;
	gate->size = NATURAL_ONE << num_qubits;
	if ((NATURAL_TYPE)PyList_Size(list) != gate->size) {
		PyErr_SetString(DokiError,"Wrong matrix size for specified number of qubits");
		free(gate);
		return NULL;
	}

	gate->matrix = MALLOC_TYPE(gate->size, COMPLEX_TYPE *);
	if (gate->matrix == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate qgate matrix");
		free(gate);
		return NULL;
	}
	
	for (i = 0; i < gate->size; i++) {
		row = PyList_GetItem(list, i);
		if (!PyList_Check(row) ||
		    (NATURAL_TYPE)PyList_Size(row) != gate->size) {
			PyErr_SetString(DokiError, "rows must be lists of size 2^num_qubits");
			for (k = 0; k < i; k++) {
				free(gate->matrix[k]);
			}
			free(gate->matrix);
			free(gate);
			return NULL;
		}
		gate->matrix[i] = MALLOC_TYPE(gate->size, COMPLEX_TYPE);
		for (j = 0; j < gate->size; j++) {
			raw_val = PyList_GetItem(row, j);
			if (PyComplex_Check(raw_val)) {
				val = COMPLEX_INIT( PyComplex_RealAsDouble(raw_val), PyComplex_ImagAsDouble(raw_val));
			} else if (PyFloat_Check(raw_val)) {
				val = COMPLEX_INIT(PyFloat_AsDouble(raw_val), 0.0);
			} else if (PyLong_Check(raw_val)) {
				val = COMPLEX_INIT(
					(double)PyLong_AsLong(raw_val), 0.0);
			} else {
				PyErr_SetString(DokiError, "matrix elements must be complex numbers");
				for (k = 0; k <= i; k++) {
					free(gate->matrix[k]);
				}
				free(gate->matrix);
				free(gate);
				return NULL;
			}
			gate->matrix[i][j] = val;
		}
	}

	return PyCapsule_New((void *)gate, "qsimov.doki.gate", &doki_gate_destroy);
}

// Revisada
static PyObject *doki_master_registry_get(PyObject *self, PyObject *args) {
	PyObject *capsule, *result;
	void *raw_master_state;
	struct master_state_vector *master_state;
	NATURAL_TYPE id;
	COMPLEX_TYPE val, aux;
	REAL_TYPE phase;
	int canonical, debug_enabled;

	if (!PyArg_ParseTuple(args, "OKpp", &capsule, &id, &canonical, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: master_registry_get(master_registry, id, canonical, verbose)");
		return NULL;
	}

	raw_master_state = PyCapsule_GetPointer(capsule, "qsimov.doki.master_state_vector");
	if (raw_master_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to master_registry");
		return NULL;
	}
	master_state = (struct master_state_vector *)raw_master_state;
	val = master_state_get(master_state, id);
	if (canonical) {
		phase = get_master_global_phase(master_state);
		aux = COMPLEX_INIT(COS(phase), -SIN(phase));
		val = COMPLEX_MULT(val, aux);
	}
	result = PyComplex_FromDoubles(RE(val), IM(val));
	return result;
}




static PyObject *doki_gate_get(PyObject *self, PyObject *args) {
	PyObject *capsule, *result, *aux;
	COMPLEX_TYPE val;
	NATURAL_TYPE i, j;
	void *raw_gate;
	struct qgate *gate;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &capsule, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: gate_get(gate, verbose)");
		return NULL;
	}

	raw_gate = PyCapsule_GetPointer(capsule, "qsimov.doki.gate");
	if (raw_gate == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to gate");
		return NULL;
	}
	gate = (struct qgate *)raw_gate;
	result = PyList_New(gate->size);
	for (i = 0; i < gate->size; i++) {
		aux = PyList_New(gate->size);
		for (j = 0; j < gate->size; j++) {
			val = gate->matrix[i][j];
			PyList_SET_ITEM(aux, j, PyComplex_FromDoubles(RE(val), IM(val)));
		}
		PyList_SET_ITEM(result, i, aux);
	}

	return result;
}

static PyObject *doki_registry_get(PyObject *self, PyObject *args) {
	PyObject *capsule, *result;
	void *raw_state;
	struct state_vector *state;
	NATURAL_TYPE id;
	COMPLEX_TYPE val, aux;
	REAL_TYPE phase;
	int canonical, debug_enabled;

	if (!PyArg_ParseTuple(args, "OKpp", &capsule, &id, &canonical, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_get(registry, id, canonical, verbose)");
		return NULL;
	}

	raw_state = PyCapsule_GetPointer(capsule, "qsimov.doki.state_vector");
	if (raw_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry");
		return NULL;
	}
	state = (struct state_vector *)raw_state;
	val = state_get(state, id);
	if (canonical) {
		phase = get_global_phase(state);
		aux = COMPLEX_INIT(COS(phase), -SIN(phase));
		val = COMPLEX_MULT(val, aux);
	}
	result = PyComplex_FromDoubles(RE(val), IM(val));

	return result;
}

void custom_state_init_py(PyObject *values, struct state_vector *state) {
	NATURAL_TYPE i;
	COMPLEX_TYPE val;
	PyObject *aux;

	for (i = 0; i < state->size; i++) {
		aux = PyList_GetItem(values, i);
		val = COMPLEX_INIT(PyComplex_RealAsDouble(aux), PyComplex_ImagAsDouble(aux));
		state_set(state, i, val);
	}
}

void custom_state_init_np(PyObject *values, struct state_vector *state) {
	NATURAL_TYPE i;
	COMPLEX_TYPE val;
	PyObject *aux;

	for (i = 0; i < state->size; i++) {
		PyArrayObject *array = (PyArrayObject *)values;
		aux = PyArray_GETITEM(array, PyArray_GETPTR1(array, i));
		val = COMPLEX_INIT(PyComplex_RealAsDouble(aux), PyComplex_ImagAsDouble(aux));
		state_set(state, i, val);
	}
}




static PyObject *doki_registry_new_data(PyObject *self, PyObject *args) {
	PyObject *raw_vals;
	unsigned int num_qubits;
	unsigned char result;
	struct state_vector *state;
	short debug_enabled;

	if (!PyArg_ParseTuple(args, "IOh", &num_qubits, &raw_vals, &debug_enabled)) {
		PyErr_SetString( DokiError, "Syntax: registry_new_data(num_qubits, values, verbose)");
		return NULL;
	}
	if (num_qubits == 0) {
		PyErr_SetString(DokiError, "num_qubits can't be zero");
		return NULL;
	}
	if (debug_enabled) {
		printf("[DEBUG] State allocation\n");
	}
	state = MALLOC_TYPE(1, struct state_vector);
	if (state == NULL) {
		PyErr_SetString(DokiError,
				"Failed to allocate state structure");
		return NULL;
	}
	if (debug_enabled) {
		printf("[DEBUG] State initialization\n");
	}
	result = state_init(state, num_qubits, false);
	if (result == 1) {
		PyErr_SetString(DokiError, "Failed to allocate state vector");
		return NULL;
	} else if (result == 2) {
		PyErr_SetString(DokiError, "Failed to allocate state chunk");
		return NULL;
	} else if (result == 3) {
		PyErr_SetString(DokiError, "Number of qubits exceeds maximum");
		return NULL;
	} else if (result != 0) {
		PyErr_SetString(DokiError, "Unknown error when creating state");
		return NULL;
	}
	if (debug_enabled) {
		printf("[DEBUG] Dumping data...\n");
	}
	if (PyArray_Check(raw_vals)) {
		if (debug_enabled) {
			printf("[DEBUG] Checking array type\n");
		}
		PyArrayObject *array = (PyArrayObject *)raw_vals;
		if (!PyArray_ISNUMBER(array)) {
			PyErr_SetString(DokiError, "values have to be numbers");
			return NULL;
		}
		if (debug_enabled) {
			printf("[DEBUG] Checking array size\n");
		}
		if (PyArray_SIZE(array) != state->size) {
			PyErr_SetString( DokiError, "Wrong array size for the specified number of qubits");
			return NULL;
		}
		if (debug_enabled) {
			printf("[DEBUG] Working with numpy array\n");
		}
		custom_state_init_np((PyObject *)array, state);
	} else if (PyList_Check(raw_vals)) {
		if (debug_enabled) {
			printf("[DEBUG] Checking list size\n");
		}
		if (PyList_GET_SIZE(raw_vals) != state->size) {
			PyErr_SetString(
				DokiError,
				"Wrong list size for the specified number of qubits\n");
			return NULL;
		}
		if (debug_enabled) {
			printf("[DEBUG] Working with python list\n");
		}
		custom_state_init_py(raw_vals, state);
	} else {
		PyErr_SetString( DokiError, "values has to be either a python list or a numpy array");
		return NULL;
	}
	if (debug_enabled) {
		printf("[DEBUG] Starting creation\n");
	}

	return PyCapsule_New((void *)state, "qsimov.doki.state_vector", &doki_registry_destroy);
}

static PyObject *doki_registry_apply(PyObject *self, PyObject *args) {
	PyObject *raw_val, *state_capsule, *gate_capsule, *target_list, *control_set, *acontrol_set, *aux;
	void *raw_state, *raw_gate;
	struct state_vector *state, *new_state;
	struct qgate *gate;
	unsigned char exit_code;
	unsigned int num_targets, num_controls, num_anticontrols, i;
	unsigned int *targets, *controls, *anticontrols;
	int num_threads, debug_enabled;

	if (!PyArg_ParseTuple(args, "OOOOOip", &state_capsule, &gate_capsule, &target_list, &control_set, &acontrol_set, &num_threads, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_apply(registry, gate, target_list, control_set, anticontrol_set, num_threads, verbose)");
		return NULL;
	}

	if (num_threads <= 0 && num_threads != -1) {
		PyErr_SetString(DokiError, "num_threads must be at least 1 (or -1 to let OpenMP choose)");
		return NULL;
	}

	raw_state = PyCapsule_GetPointer(state_capsule, "qsimov.doki.state_vector");
	if (raw_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry");
		return NULL;
	}
	state = (struct state_vector *)raw_state;

	raw_gate = PyCapsule_GetPointer(gate_capsule, "qsimov.doki.gate");
	if (raw_gate == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to gate");
		return NULL;
	}
	gate = (struct qgate *)raw_gate;

	if (!PyList_Check(target_list)) {
		PyErr_SetString(DokiError, "target_list must be a list");
		return NULL;
	}

	num_targets = (unsigned int)PyList_Size(target_list);
	if (num_targets != gate->num_qubits) {
		PyErr_SetString(
			DokiError,
			"Wrong number of targets specified for that gate");
		return NULL;
	}

	num_controls = 0;
	if (PySet_Check(control_set)) {
		num_controls = (unsigned int)PySet_Size(control_set);
	} else if (control_set != Py_None) {
		PyErr_SetString(DokiError, "control_set must be a set or None");
		return NULL;
	}

	num_anticontrols = 0;
	if (PySet_Check(acontrol_set)) {
		num_anticontrols = (unsigned int)PySet_Size(acontrol_set);
	} else if (acontrol_set != Py_None) {
		PyErr_SetString(DokiError,
				"anticontrol_set must be a set or None");
		return NULL;
	}

	targets = MALLOC_TYPE(num_targets, unsigned int);
	if (targets == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate target array");
		return NULL;
	}
	controls = NULL;
	if (num_controls > 0) {
		controls = MALLOC_TYPE(num_controls, unsigned int);
		if (controls == NULL) {
			PyErr_SetString(DokiError,
					"Failed to allocate control array");
			return NULL;
		}
	}
	anticontrols = NULL;
	if (num_anticontrols > 0) {
		anticontrols = MALLOC_TYPE(num_anticontrols, unsigned int);
		if (anticontrols == NULL) {
			PyErr_SetString(DokiError,
					"Failed to allocate anticontrol array");
			return NULL;
		}
	}

	if (num_controls > 0) {
		aux = PySet_New(control_set);
		for (i = 0; i < num_controls; i++) {
			raw_val = PySet_Pop(aux);
			if (!PyLong_Check(raw_val)) {
				PyErr_SetString(
					DokiError,
					"control_set must be a set qubit ids (unsigned integers)");
				return NULL;
			}
			controls[i] = PyLong_AsLong(raw_val);
			if (controls[i] >= state->num_qubits) {
				PyErr_SetString(DokiError,
						"Control qubit out of range");
				return NULL;
			}
		}
	}

	if (num_anticontrols > 0) {
		aux = PySet_New(acontrol_set);
		for (i = 0; i < num_anticontrols; i++) {
			raw_val = PySet_Pop(aux);
			if (!PyLong_Check(raw_val)) {
				PyErr_SetString(
					DokiError,
					"anticontrol_set must be a set "
					"qubit ids (unsigned integers)");
				return NULL;
			}
			if (PySet_Contains(control_set, raw_val)) {
				PyErr_SetString(
					DokiError,
					"A control cannot also be an anticontrol");
				return NULL;
			}
			anticontrols[i] = PyLong_AsLong(raw_val);
			if (anticontrols[i] >= state->num_qubits) {
				PyErr_SetString(
					DokiError,
					"Anticontrol qubit out of range");
				return NULL;
			}
		}
	}

	for (i = 0; i < num_targets; i++) {
		raw_val = PyList_GetItem(target_list, i);
		if (!PyLong_Check(raw_val)) {
			PyErr_SetString(
				DokiError,
				"target_list must be a list of qubit ids (unsigned integers)");
			return NULL;
		}
		if ((num_controls > 0 &&
		     PySet_Contains(control_set, raw_val)) ||
		    (num_anticontrols > 0 &&
		     PySet_Contains(acontrol_set, raw_val))) {
			PyErr_SetString(
				DokiError,
				"A target cannot also be a control or an anticontrol");
			return NULL;
		}
		targets[i] = PyLong_AsLong(raw_val);
		if (targets[i] >= state->num_qubits) {
			PyErr_SetString(DokiError, "Target qubit out of range");
			return NULL;
		}
	}

	new_state = MALLOC_TYPE(1, struct state_vector);
	if (new_state == NULL) {
		PyErr_SetString(DokiError,
				"Failed to allocate new state structure");
		return NULL;
	}
	if (num_threads != -1) {
		omp_set_num_threads(num_threads);
	}

	exit_code = apply_gate(state, gate, targets, num_targets, controls, num_controls, anticontrols, num_anticontrols, new_state);

	if (exit_code == 1) {
		PyErr_SetString(DokiError,
				"Failed to initialize new state chunk");
	} else if (exit_code == 2) {
		PyErr_SetString(DokiError,
				"Failed to allocate new state chunk");
	} else if (exit_code == 3) {
		PyErr_SetString(
			DokiError,
			"[BUG] THIS SHOULD NOT HAPPEN. Failed to set first value to 1");
	} else if (exit_code == 4) {
		PyErr_SetString(
			DokiError,
			"Failed to allocate new state vector structure");
	} else if (exit_code == 5) {
		PyErr_SetString(DokiError, "Failed to apply gate");
	} else if (exit_code == 11) {
		PyErr_SetString(DokiError,
				"Failed to allocate not_copy structure");
	} else if (exit_code != 0) {
		PyErr_SetString(DokiError, "Unknown error when applying gate");
	}

	if (exit_code > 0) {
		free(targets);
		if (num_controls > 0) {
			free(controls);
		}
		if (num_anticontrols > 0) {
			free(anticontrols);
		}
		return NULL;
	}

	return PyCapsule_New((void *)new_state, "qsimov.doki.state_vector",
			     &doki_registry_destroy);
}

static PyObject *doki_registry_join(PyObject *self, PyObject *args) {
	PyObject *capsule1, *capsule2;
	void *raw_state1, *raw_state2;
	struct state_vector *state1, *state2, *result;
	unsigned char exit_code;
	int num_threads, debug_enabled;

	if (!PyArg_ParseTuple(args, "OOip", &capsule1, &capsule2, &num_threads,
			      &debug_enabled)) {
		PyErr_SetString(DokiError,
				"Syntax: registry_join(most_registry, "
				"least_registry, num_threads, verbose)");
		return NULL;
	}

	if (num_threads <= 0 && num_threads != -1) {
		PyErr_SetString(
			DokiError,
			"num_threads must be at least 1 (or -1 to let OpenMP choose)");
		return NULL;
	}

	raw_state1 = PyCapsule_GetPointer(capsule1, "qsimov.doki.state_vector");
	if (raw_state1 == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry1");
		return NULL;
	}

	raw_state2 = PyCapsule_GetPointer(capsule2, "qsimov.doki.state_vector");
	if (raw_state2 == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry2");
		return NULL;
	}
	state1 = (struct state_vector *)raw_state1;
	state2 = (struct state_vector *)raw_state2;
	result = MALLOC_TYPE(1, struct state_vector);
	if (result == NULL) {
		PyErr_SetString(DokiError,
				"Failed to allocate new state structure");
		return NULL;
	}
	if (num_threads != -1) {
		omp_set_num_threads(num_threads);
	}
	exit_code = join(result, state1, state2);
	if (exit_code != 0) {
		switch (exit_code) {
		case 1:
			PyErr_SetString(DokiError,
					"Failed to initialize new state chunk");
			break;
		case 2:
			PyErr_SetString(DokiError,
					"Failed to allocate new state chunk");
			break;
		case 3:
			PyErr_SetString(
				DokiError,
				"[BUG] THIS SHOULD NOT HAPPEN. Failed to set first value to 1");
			break;
		case 4:
			PyErr_SetString(
				DokiError,
				"Failed to allocate new state vector structure");
			break;
		case 5:
			PyErr_SetString(DokiError, "Failed to get/set a value");
			break;
		default:
			PyErr_SetString(DokiError,
					"Unknown error when joining states");
		}
		return NULL;
	}

	return PyCapsule_New((void *)result, "qsimov.doki.state_vector",
			     &doki_registry_destroy);
}

static PyObject *doki_registry_measure(PyObject *self, PyObject *args) {
	PyObject *capsule, *py_measured_val, *result, *new_capsule, *roll_list;
	Py_ssize_t roll_id;
	void *raw_state;
	struct state_vector *state, *new_state, *aux;
	NATURAL_TYPE mask;
	REAL_TYPE roll;
	unsigned int i, curr_id, initial_num_qubits, measured_qty;
	_Bool measure_id, measured_val;
	unsigned char exit_code;
	int debug_enabled, num_threads;

	if (!PyArg_ParseTuple(args, "OKOip", &capsule, &mask, &roll_list, &num_threads, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_measure(registry, mask, roll_list, num_threads, verbose)");
		return NULL;
	}

	if (num_threads <= 0 && num_threads != -1) {
		PyErr_SetString(
			DokiError,
			"num_threads must be at least 1 (or -1 to let OpenMP choose)");
		return NULL;
	}

	raw_state = PyCapsule_GetPointer(capsule, "qsimov.doki.state_vector");
	if (raw_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry");
		return NULL;
	}
	if (!PyList_Check(roll_list)) {
		PyErr_SetString(DokiError, "roll_list must be a list of real numbers in [0, 1)!");
		return NULL;
	}
	state = (struct state_vector *)raw_state;
	initial_num_qubits = state->num_qubits;
	result = PyList_New(initial_num_qubits);

	new_state = MALLOC_TYPE(1, struct state_vector);
	if (new_state == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate new state structure");
		return NULL;
	}

	if (num_threads != -1) {
		omp_set_num_threads(num_threads);
	}
	exit_code = state_clone(new_state, state);
	if (exit_code == 1) {
		PyErr_SetString(DokiError, "Failed to allocate state vector");
		return NULL;
	} else if (exit_code == 2) {
		PyErr_SetString(DokiError, "Failed to allocate state chunk");
		return NULL;
	} else if (exit_code == 3) {
		if (debug_enabled) {
			printf("[DEBUG] %u", state->num_qubits);
		}
		PyErr_SetString(DokiError, "Wrong number of qubits");
		return NULL;
	} else if (exit_code != 0) {
		PyErr_SetString(DokiError, "Unknown error when cloning state");
		return NULL;
	}

	measured_qty = 0;
	roll_id = 0;
	aux = NULL;
	for (i = 0; i < initial_num_qubits; i++) {
		curr_id = initial_num_qubits - i - 1;
		measure_id = mask & (NATURAL_ONE << curr_id);
		py_measured_val = Py_None;
		if (measure_id) {
			if (new_state == NULL || new_state->num_qubits == 0) {
				if (new_state != NULL) {
					state_clear(new_state);
					free(new_state);
				}
				PyErr_SetString(
					DokiError,
					"Could not measure non_existant qubits");
				return NULL;
			}
			roll = PyFloat_AsDouble(
				PyList_GetItem(roll_list, roll_id));
			if (roll < 0 || roll >= 1) {
				state_clear(new_state);
				free(new_state);
				PyErr_SetString(DokiError,
						"roll not in interval [0, 1)!");
				return NULL;
			}
			roll_id++;
			aux = MALLOC_TYPE(1, struct state_vector);
			if (aux == NULL) {
				state_clear(new_state);
				free(new_state);
				PyErr_SetString(
					DokiError,
					"Failed to allocate aux state structure");
				return NULL;
			}
			exit_code = measure(new_state, &measured_val, curr_id,
					    aux, roll);
			if (exit_code != 0) {
				state_clear(aux);
				free(aux);
				aux = NULL;
				break;
			}
			if (aux->num_qubits > 0 && aux->norm_const == 0.0) {
				state_clear(aux);
				free(aux);
				state_clear(new_state);
				free(new_state);
				PyErr_SetString(
					DokiError,
					"New normalization constant is 0. Please report "
					"this error with the steps to reproduce it.");
				return NULL;
			}
			measured_qty++;
			py_measured_val = measured_val ? Py_True : Py_False;
			state_clear(new_state);
			free(new_state);
			new_state = aux;
			aux = NULL;
		}
		PyList_SET_ITEM(result, i, py_measured_val);
	}
	if (exit_code != 0) {
		if (new_state != NULL) {
			state_clear(new_state);
			free(new_state);
		}
		switch (exit_code) {
		case 1:
			PyErr_SetString(DokiError,
					"Failed to allocate state vector");
			break;
		case 2:
			PyErr_SetString(DokiError,
					"Failed to allocate state chunk");
			break;
		default:
			PyErr_SetString(DokiError,
					"Unknown error while collapsing state");
		}
		return NULL;
	}

	if (state->num_qubits - measured_qty > 0) {
		new_capsule = PyCapsule_New((void *)new_state,
					    "qsimov.doki.state_vector",
					    &doki_registry_destroy);
	} else {
		if (new_state != NULL) {
			state_clear(new_state);
			free(new_state);
		}
		new_capsule = Py_None;
	}
	return PyTuple_Pack(2, new_capsule, result);
}

static PyObject *doki_registry_prob(PyObject *self, PyObject *args) {
	PyObject *capsule;
	void *raw_state;
	struct state_vector *state;
	unsigned int id;
	int debug_enabled, num_threads;

	if (!PyArg_ParseTuple(args, "OIip", &capsule, &id, &num_threads,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: registry_prob(registry, qubit_id, num_threads, verbose)");
		return NULL;
	}

	if (num_threads <= 0 && num_threads != -1) {
		PyErr_SetString(
			DokiError,
			"num_threads must be at least 1 (or -1 to let OpenMP choose)");
		return NULL;
	}

	raw_state = PyCapsule_GetPointer(capsule, "qsimov.doki.state_vector");
	if (raw_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry");
		return NULL;
	}
	state = (struct state_vector *)raw_state;

	if (num_threads != -1) {
		omp_set_num_threads(num_threads);
	}
	return PyFloat_FromDouble(probability(state, id));
}

static PyObject *doki_registry_density(PyObject *self, PyObject *args) {
	PyObject *state_capsule;
	void *raw_state;
	struct FMatrix *densityMatrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &state_capsule, &debug_enabled)) {
		PyErr_SetString(DokiError, "Syntax: registry_density(state, verbose)");
		return NULL;
	}

	raw_state = PyCapsule_GetPointer(state_capsule, "qsimov.doki.state_vector");
	if (raw_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry");
		return NULL;
	}

	densityMatrix = density_matrix(state_capsule);
	if (densityMatrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(DokiError, "[DENSITY] Failed to allocate density matrix");
			break;
		case 2:
			PyErr_SetString(DokiError, "[DENSITY] The state is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[DENSITY] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New((void *)densityMatrix, "qsimov.doki.funmatrix", &doki_funmatrix_destroy);
}

static PyObject *doki_registry_mem(PyObject *self, PyObject *args) {
	PyObject *state_capsule;
	void *raw_state;
	size_t size;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &state_capsule, &debug_enabled)) {
		PyErr_SetString(DokiError,
				"Syntax: registry_mem(state, verbose)");
		return NULL;
	}

	raw_state =
		PyCapsule_GetPointer(state_capsule, "qsimov.doki.state_vector");
	if (raw_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry");
		return NULL;
	}

	size = state_mem_size((struct state_vector *)raw_state);

	return PyLong_FromSize_t(size);
}

static PyObject *doki_funmatrix_create(PyObject *self, PyObject *args) {
	PyObject *list, *row, *raw_val;
	Py_ssize_t num_rows, num_cols;
	NATURAL_TYPE size, i, j;
	COMPLEX_TYPE val, *matrix_2d;
	struct FMatrix *funmatrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &list, &debug_enabled)) {
		PyErr_SetString(DokiError,
				"Syntax: funmatrix_create(matrix, verbose)");
		return NULL;
	}
	if (!PyList_Check(list)) {
		PyErr_SetString(DokiError,
				"matrix must be a list of lists (matrix)");
		return NULL;
	}

	num_rows = PyList_Size(list);
	if (num_rows == 0) {
		PyErr_SetString(DokiError, "there are no rows");
		return NULL;
	}
	row = PyList_GetItem(list, 0);
	if (!PyList_Check(row)) {
		PyErr_SetString(DokiError, "rows must be lists");
		return NULL;
	}
	num_cols = PyList_Size(row);
	if (num_cols == 0) {
		PyErr_SetString(DokiError, "there are no columns");
		return NULL;
	}

	size = (NATURAL_TYPE)num_rows * num_cols;
	matrix_2d = MALLOC_TYPE(size, COMPLEX_TYPE);
	if (matrix_2d == NULL) {
		PyErr_SetString(DokiError, "failed to allocate 2D matrix");
		return NULL;
	}

	for (i = 0; i < num_rows; ++i) {
		row = PyList_GetItem(list, i);
		if (!PyList_Check(row) ||
		    (NATURAL_TYPE)PyList_Size(row) != num_cols) {
			PyErr_SetString(DokiError,
					"rows must be lists of the same size");
			free(matrix_2d);
			return NULL;
		}
		for (j = 0; j < num_cols; ++j) {
			raw_val = PyList_GetItem(row, j);
			if (PyComplex_Check(raw_val)) {
				val = COMPLEX_INIT(
					PyComplex_RealAsDouble(raw_val),
					PyComplex_ImagAsDouble(raw_val));
			} else if (PyFloat_Check(raw_val)) {
				val = COMPLEX_INIT(PyFloat_AsDouble(raw_val),
						   0.0);
			} else if (PyLong_Check(raw_val)) {
				val = COMPLEX_INIT(
					(double)PyLong_AsLong(raw_val), 0.0);
			} else {
				PyErr_SetString(
					DokiError,
					"matrix elements must be complex numbers");
				free(matrix_2d);
				return NULL;
			}
			matrix_2d[i * num_rows + j] = val;
		}
	}
	funmatrix = CustomMat(matrix_2d, size, num_rows, num_cols);
	if (funmatrix == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to matrix");
		return NULL;
	}

	return PyCapsule_New((void *)funmatrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_identity(PyObject *self, PyObject *args) {
	unsigned int num_qubits;
	struct FMatrix *funmatrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Ip", &num_qubits, &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_identity(num_qubits, verbose)");
		return NULL;
	}
	funmatrix = Identity(num_qubits);

	return PyCapsule_New((void *)funmatrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_hadamard(PyObject *self, PyObject *args) {
	unsigned int num_qubits;
	struct FMatrix *funmatrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Ip", &num_qubits, &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_hadamard(num_qubits, verbose)");
		return NULL;
	}
	funmatrix = Hadamard(num_qubits);
	if (funmatrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(DokiError,
					"[H] Failed to allocate result matrix");
			break;
		case 5:
			PyErr_SetString(DokiError,
					"[H] Failed to allocate data pointer");
			break;
		default:
			PyErr_SetString(DokiError, "[H] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New((void *)funmatrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_statezero(PyObject *self, PyObject *args) {
	unsigned int num_qubits;
	struct FMatrix *funmatrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Ip", &num_qubits, &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_statezero(num_qubits, verbose)");
		return NULL;
	}
	funmatrix = StateZero(num_qubits);
	if (funmatrix == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to matrix");
		return NULL;
	}

	return PyCapsule_New((void *)funmatrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_densityzero(PyObject *self, PyObject *args) {
	unsigned int num_qubits;
	struct FMatrix *funmatrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Ip", &num_qubits, &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_densityzero(num_qubits, verbose)");
		return NULL;
	}
	funmatrix = DensityZero(num_qubits);
	if (funmatrix == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to matrix");
		return NULL;
	}

	return PyCapsule_New((void *)funmatrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_addcontrol(PyObject *self, PyObject *args) {
	PyObject *capsule;
	void *funmatrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &capsule, &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_addcontrol(funmatrix, verbose)");
		return NULL;
	}

	funmatrix = (void *)CU(capsule);
	if (funmatrix == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to matrix");
		return NULL;
	}

	return PyCapsule_New((void *)funmatrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_get(PyObject *self, PyObject *args) {
	PyObject *capsule;
	void *raw_matrix;
	struct FMatrix *matrix;
	NATURAL_TYPE i, j;
	COMPLEX_TYPE val;
	int debug_enabled, res;

	if (!PyArg_ParseTuple(args, "OKKp", &capsule, &i, &j, &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_get(funmatrix, i, j, verbose)");
		return NULL;
	}

	raw_matrix = PyCapsule_GetPointer(capsule, "qsimov.doki.funmatrix");
	if (raw_matrix == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to matrix");
		return NULL;
	}
	matrix = (struct FMatrix *)raw_matrix;

	if (i < 0 || j < 0 || i >= matrix->r || j >= matrix->c) {
		PyErr_SetString(DokiError, "Out of bounds");
		return NULL;
	}

	val = COMPLEX_ZERO;
	res = getitem(matrix, i, j, &val);
	if (res != 0) {
		switch (res) {
		case 1:
			PyErr_SetString(DokiError,
					"[GET] Error adding parent matrices");
			break;
		case 2:
			PyErr_SetString(
				DokiError,
				"[GET] Error substracting parent matrices");
			break;
		case 3:
			PyErr_SetString(
				DokiError,
				"[GET] Error multiplying parent matrices");
			break;
		case 4:
			PyErr_SetString(
				DokiError,
				"[GET] Error multiplying entity-wise parent matrices");
			break;
		case 5:
			PyErr_SetString(
				DokiError,
				"[GET] Error calculating Kronecker product of parent matrices");
			break;
		case 6:
			PyErr_SetString(
				DokiError,
				"[GET] Unknown operation between parent matrices");
			break;
		case 7:
			PyErr_SetString(DokiError,
					"[GET] Element out of bounds");
			break;
		case 8:
			PyErr_SetString(DokiError, "[GET] f returned NAN");
			break;
		default:
			PyErr_SetString(DokiError, "[GET] Unknown error code");
		}
		return NULL;
	}

	if (isnan(RE(val)) || isnan(IM(val))) {
		PyErr_SetString(DokiError, "[GET] Unexpected NAN value");
		return NULL;
	}

	return PyComplex_FromDoubles(RE(val), IM(val));
}

static PyObject *doki_funmatrix_add(PyObject *self, PyObject *args) {
	PyObject *capsule1, *capsule2;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OOp", &capsule1, &capsule2,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_add(funmatrix1, funmatrix2, verbose)");
		return NULL;
	}

	raw_matrix = (void *)madd(capsule1, capsule2);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[ADD] Failed to allocate result matrix");
			break;
		case 2:
			PyErr_SetString(DokiError,
					"[ADD] The operands are misalligned");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[ADD] The first operand is NULL");
			break;
		case 4:
			PyErr_SetString(DokiError,
					"[ADD] The second operand is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[ADD] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_sub(PyObject *self, PyObject *args) {
	PyObject *capsule1, *capsule2;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OOp", &capsule1, &capsule2,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_sub(funmatrix1, funmatrix2, verbose)");
		return NULL;
	}

	raw_matrix = (void *)msub(capsule1, capsule2);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[SUB] Failed to allocate result matrix");
			break;
		case 2:
			PyErr_SetString(DokiError,
					"[SUB] The operands are misalligned");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[SUB] The first operand is NULL");
			break;
		case 4:
			PyErr_SetString(DokiError,
					"[SUB] The second operand is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[SUB] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_scalar_mul(PyObject *self, PyObject *args) {
	PyObject *capsule, *raw_scalar;
	void *raw_matrix;
	COMPLEX_TYPE scalar;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OOp", &capsule, &raw_scalar,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_mul(funmatrix, scalar, verbose)");
		return NULL;
	}

	if (PyComplex_Check(raw_scalar)) {
		scalar = COMPLEX_INIT(PyComplex_RealAsDouble(raw_scalar),
				      PyComplex_ImagAsDouble(raw_scalar));
	} else if (PyFloat_Check(raw_scalar)) {
		scalar = COMPLEX_INIT(PyFloat_AsDouble(raw_scalar), 0.0);
	} else if (PyLong_Check(raw_scalar)) {
		scalar = COMPLEX_INIT((double)PyLong_AsLong(raw_scalar), 0.0);
	} else {
		PyErr_SetString(DokiError, "scalar is not a number");
		return NULL;
	}

	raw_matrix = (void *)mprod(scalar, capsule);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[SPROD] Failed to allocate result matrix");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[SPROD] The matrix operand is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[SPROD] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_scalar_div(PyObject *self, PyObject *args) {
	PyObject *capsule, *raw_scalar;
	void *raw_matrix;
	COMPLEX_TYPE scalar;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OOp", &capsule, &raw_scalar,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_div(funmatrix, scalar, verbose)");
		return NULL;
	}

	if (PyComplex_Check(raw_scalar)) {
		scalar = COMPLEX_INIT(PyComplex_RealAsDouble(raw_scalar),
				      PyComplex_ImagAsDouble(raw_scalar));
	} else if (PyFloat_Check(raw_scalar)) {
		scalar = COMPLEX_INIT(PyFloat_AsDouble(raw_scalar), 0.0);
	} else if (PyLong_Check(raw_scalar)) {
		scalar = COMPLEX_INIT((double)PyLong_AsLong(raw_scalar), 0.0);
	} else {
		PyErr_SetString(DokiError, "scalar is not a number");
		return NULL;
	}

	if (RE(scalar) == 0 && IM(scalar) == 0) {
		PyErr_SetString(DokiError, "Dividing by zero");
		return NULL;
	}
	raw_matrix = (void *)mdiv(scalar, capsule);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[SDIV] Failed to allocate result matrix");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[SDIV] The matrix operand is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[SDIV] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_matmul(PyObject *self, PyObject *args) {
	PyObject *capsule1, *capsule2;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OOp", &capsule1, &capsule2,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_matmul(funmatrix1, funmatrix2, verbose)");
		return NULL;
	}

	raw_matrix = (void *)matmul(capsule1, capsule2);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[MATMUL] Failed to allocate result matrix");
			break;
		case 2:
			PyErr_SetString(
				DokiError,
				"[MATMUL] The operands are misalligned");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[MATMUL] The first operand is NULL");
			break;
		case 4:
			PyErr_SetString(DokiError,
					"[MATMUL] The second operand is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[MATMUL] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_ewmul(PyObject *self, PyObject *args) {
	PyObject *capsule1, *capsule2;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OOp", &capsule1, &capsule2,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_ewmul(funmatrix1, funmatrix2, verbose)");
		return NULL;
	}

	raw_matrix = (void *)ewmul(capsule1, capsule2);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[EWMUL] Failed to allocate result matrix");
			break;
		case 2:
			PyErr_SetString(DokiError,
					"[EWMUL] The operands are misalligned");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[EWMUL] The first operand is NULL");
			break;
		case 4:
			PyErr_SetString(DokiError,
					"[EWMUL] The second operand is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[EWMUL] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_kron(PyObject *self, PyObject *args) {
	PyObject *capsule1, *capsule2;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OOp", &capsule1, &capsule2,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_kron(funmatrix1, funmatrix2, verbose)");
		return NULL;
	}

	raw_matrix = (void *)kron(capsule1, capsule2);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[KRON] Failed to allocate result matrix");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[KRON] The first operand is NULL");
			break;
		case 4:
			PyErr_SetString(DokiError,
					"[KRON] The second operand is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[KRON] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_eyekron(PyObject *self, PyObject *args) {
	PyObject *capsule;
	unsigned int left, right;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OIIp", &capsule, &left, &right,
			      &debug_enabled)) {
		PyErr_SetString(DokiError,
				"Syntax: funmatrix_eyekron(funmatrix, "
				"leftQubits, rightQubits, verbose)");
		return NULL;
	}

	raw_matrix = (void *)eyeKron(capsule, left, right);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[EYEKRON] Failed to allocate result matrix");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[EYEKRON] The matrix is NULL");
			break;
		case 5:
			PyErr_SetString(
				DokiError,
				"[EYEKRON] Could not allocate data array");
			break;
		case 6:
			PyErr_SetString(
				DokiError,
				"[EYEKRON] Could not allocate data struct");
			break;
		default:
			PyErr_SetString(DokiError, "[EYEKRON] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_transpose(PyObject *self, PyObject *args) {
	PyObject *capsule;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &capsule, &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_transpose(funmatrix, verbose)");
		return NULL;
	}

	raw_matrix = (void *)transpose(capsule);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[TRANS] Failed to allocate result matrix");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[TRANS] The matrix is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[TRANS] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_dagger(PyObject *self, PyObject *args) {
	PyObject *capsule;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &capsule, &debug_enabled)) {
		PyErr_SetString(DokiError,
				"Syntax: funmatrix_dagger(funmatrix, verbose)");
		return NULL;
	}

	raw_matrix = (void *)dagger(capsule);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[HTRANS] Failed to allocate result matrix");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[HTRANS] The matrix is NULL");
			break;
		default:
			PyErr_SetString(DokiError, "[HTRANS] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_projection(PyObject *self, PyObject *args) {
	PyObject *capsule;
	unsigned int qubitId;
	void *raw_matrix;
	bool value, debug_enabled;

	if (!PyArg_ParseTuple(args, "OIpp", &capsule, &qubitId, &value,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_projection(funmatrix, qubit_id, value, verbose)");
		return NULL;
	}

	raw_matrix = (void *)projection(capsule, qubitId, value);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[PROJ] Failed to allocate result matrix");
			break;
		case 3:
			PyErr_SetString(DokiError, "[PROJ] The matrix is NULL");
			break;
		case 5:
			PyErr_SetString(
				DokiError,
				"[PROJ] Could not allocate data struct");
			break;
		default:
			PyErr_SetString(DokiError, "[PROJ] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_shape(PyObject *self, PyObject *args) {
	PyObject *capsule;
	void *raw_matrix;
	struct FMatrix *matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &capsule, &debug_enabled)) {
		PyErr_SetString(DokiError,
				"Syntax: funmatrix_shape(funmatrix, verbose)");
		return NULL;
	}

	raw_matrix = PyCapsule_GetPointer(capsule, "qsimov.doki.funmatrix");
	if (raw_matrix == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to matrix");
		return NULL;
	}
	matrix = (struct FMatrix *)raw_matrix;

	return PyTuple_Pack(2, PyLong_FromUnsignedLongLong(rows(matrix)),
			    PyLong_FromUnsignedLongLong(columns(matrix)));
}

static PyObject *doki_funmatrix_partialtrace(PyObject *self, PyObject *args) {
	PyObject *capsule;
	unsigned int id;
	void *raw_matrix;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "OIp", &capsule, &id, &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_partialtrace(funmatrix, id, verbose)");
		return NULL;
	}

	raw_matrix = (void *)partial_trace(capsule, id);
	if (raw_matrix == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(
				DokiError,
				"[PTRACE] Failed to allocate result matrix");
			break;
		case 2:
			PyErr_SetString(
				DokiError,
				"[PTRACE] The matrix is not a square matrix");
			break;
		case 3:
			PyErr_SetString(DokiError,
					"[PTRACE] The matrix is NULL");
			break;
		case 5:
			PyErr_SetString(
				DokiError,
				"[PTRACE] Could not allocate argv struct");
			break;
		case 6:
			PyErr_SetString(DokiError,
					"[PTRACE] elem id has to be >= 0");
			break;
		default:
			PyErr_SetString(DokiError, "[PTRACE] Unknown error");
		}
		return NULL;
	}

	return PyCapsule_New(raw_matrix, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_trace(PyObject *self, PyObject *args) {
	PyObject *capsule;
	void *raw_matrix;
	struct FMatrix *matrix;
	COMPLEX_TYPE result, aux;
	NATURAL_TYPE i, min_shape;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &capsule, &debug_enabled)) {
		PyErr_SetString(DokiError,
				"Syntax: funmatrix_trace(funmatrix, verbose)");
		return NULL;
	}
	raw_matrix = PyCapsule_GetPointer(capsule, "qsimov.doki.funmatrix");
	if (raw_matrix == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to matrix");
		return NULL;
	}
	matrix = (struct FMatrix *)raw_matrix;
	result = COMPLEX_ZERO;
	min_shape = matrix->r <= matrix->c ? matrix->r : matrix->c;

	aux = COMPLEX_ZERO;
	for (i = 0; i < min_shape; i++) {
		int res = getitem(matrix, i, i, &aux);
		if (res != 0) {
			switch (res) {
			case 1:
				PyErr_SetString(
					DokiError,
					"[TRACE] Error adding parent matrices");
				break;
			case 2:
				PyErr_SetString(
					DokiError,
					"[TRACE] Error substracting parent matrices");
				break;
			case 3:
				PyErr_SetString(
					DokiError,
					"[TRACE] Error multiplying parent matrices");
				break;
			case 4:
				PyErr_SetString(
					DokiError,
					"[TRACE] Error multiplying entity-wise parent matrices");
				break;
			case 5:
				PyErr_SetString(
					DokiError,
					"[TRACE] Error calculating Kronecker product "
					"of parent matrices");
				break;
			case 6:
				PyErr_SetString(
					DokiError,
					"[TRACE] Unknown operation between parent matrices");
				break;
			case 7:
				PyErr_SetString(
					DokiError,
					"[TRACE] Element out of bounds");
				break;
			case 8:
				PyErr_SetString(DokiError,
						"[TRACE] f returned NAN");
				break;
			default:
				PyErr_SetString(DokiError,
						"[TRACE] Unknown error code");
			}
			return NULL;
		}

		if (isnan(RE(aux)) || isnan(IM(aux))) {
			PyErr_SetString(DokiError,
					"[TRACE] Unexpected NAN value");
			return NULL;
		}
		result = COMPLEX_ADD(result, aux);
	}

	return PyComplex_FromDoubles(RE(result), IM(result));
}

static PyObject *doki_funmatrix_apply(PyObject *self, PyObject *args) {
	PyObject *raw_val, *state_capsule, *gate_capsule, *target_list,
		*control_set, *acontrol_set, *aux;
	void *raw_state, *raw_gate;
	struct FMatrix *state, *new_state, *gate;
	unsigned int num_targets, num_controls, num_anticontrols, num_qubits,
		num_qb_gate, i;
	unsigned int *targets, *controls, *anticontrols;
	bool debug_enabled;

	if (!PyArg_ParseTuple(args, "OOOOOp", &state_capsule, &gate_capsule,
			      &target_list, &control_set, &acontrol_set,
			      &debug_enabled)) {
		PyErr_SetString(
			DokiError,
			"Syntax: funmatrix_apply(registry, gate, target_list, "
			"control_set, anticontrol_set, verbose)");
		return NULL;
	}

	raw_state =
		PyCapsule_GetPointer(state_capsule, "qsimov.doki.funmatrix");
	if (raw_state == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to registry");
		return NULL;
	}
	state = (struct FMatrix *)raw_state;

	raw_gate = PyCapsule_GetPointer(gate_capsule, "qsimov.doki.funmatrix");
	if (raw_gate == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to gate");
		return NULL;
	}
	gate = (struct FMatrix *)raw_gate;

	if (state->c > 1) {
		PyErr_SetString(DokiError, "registry is not a column vector");
		return NULL;
	}

	if (gate->c != gate->r) {
		PyErr_SetString(DokiError, "gates have to be square matrices");
		return NULL;
	}

	if (gate->r > state->r) {
		PyErr_SetString(DokiError,
				"gate is too big for this state vector");
		return NULL;
	}

	num_qubits = log2_64(state->r);
	if (state->r != NATURAL_ONE << num_qubits) {
		PyErr_SetString(DokiError, "registry needs 2^n rows");
		return NULL;
	}

	num_qb_gate = log2_64(gate->r);
	if (gate->r != NATURAL_ONE << num_qb_gate) {
		PyErr_SetString(DokiError, "gates need 2^n x 2^n elements");
		return NULL;
	}

	if (!PyList_Check(target_list)) {
		PyErr_SetString(DokiError, "target_list must be a list");
		return NULL;
	}

	num_targets = (unsigned int)PyList_Size(target_list);
	if (num_targets != num_qb_gate) {
		PyErr_SetString(
			DokiError,
			"Wrong number of targets specified for that gate");
		return NULL;
	}

	num_controls = 0;
	if (PySet_Check(control_set)) {
		num_controls = (unsigned int)PySet_Size(control_set);
	} else if (control_set != Py_None) {
		PyErr_SetString(DokiError, "control_set must be a set or None");
		return NULL;
	}

	num_anticontrols = 0;
	if (PySet_Check(acontrol_set)) {
		num_anticontrols = (unsigned int)PySet_Size(acontrol_set);
	} else if (acontrol_set != Py_None) {
		PyErr_SetString(DokiError,
				"anticontrol_set must be a set or None");
		return NULL;
	}

	targets = MALLOC_TYPE(num_targets, unsigned int);
	if (targets == NULL) {
		PyErr_SetString(DokiError, "Failed to allocate target array");
		return NULL;
	}
	controls = NULL;
	if (num_controls > 0) {
		controls = MALLOC_TYPE(num_controls, unsigned int);
		if (controls == NULL) {
			PyErr_SetString(DokiError,
					"Failed to allocate control array");
			return NULL;
		}
	}
	anticontrols = NULL;
	if (num_anticontrols > 0) {
		anticontrols = MALLOC_TYPE(num_anticontrols, unsigned int);
		if (anticontrols == NULL) {
			PyErr_SetString(DokiError,
					"Failed to allocate anticontrol array");
			return NULL;
		}
	}

	if (num_controls > 0) {
		aux = PySet_New(control_set);
		for (i = 0; i < num_controls; i++) {
			raw_val = PySet_Pop(aux);
			if (!PyLong_Check(raw_val)) {
				PyErr_SetString(
					DokiError,
					"control_set must be a set qubit ids (unsigned integers)");
				return NULL;
			}
			controls[i] = PyLong_AsLong(raw_val);
			if (controls[i] >= num_qubits) {
				PyErr_SetString(DokiError,
						"Control qubit out of range");
				return NULL;
			}
		}
	}

	if (num_anticontrols > 0) {
		aux = PySet_New(acontrol_set);
		for (i = 0; i < num_anticontrols; i++) {
			raw_val = PySet_Pop(aux);
			if (!PyLong_Check(raw_val)) {
				PyErr_SetString(
					DokiError,
					"anticontrol_set must be a set "
					"qubit ids (unsigned integers)");
				return NULL;
			}
			if (PySet_Contains(control_set, raw_val)) {
				PyErr_SetString(
					DokiError,
					"A control cannot also be an anticontrol");
				return NULL;
			}
			anticontrols[i] = PyLong_AsLong(raw_val);
			if (anticontrols[i] >= num_qubits) {
				PyErr_SetString(
					DokiError,
					"Anticontrol qubit out of range");
				return NULL;
			}
		}
	}

	for (i = 0; i < num_targets; i++) {
		raw_val = PyList_GetItem(target_list, i);
		if (!PyLong_Check(raw_val)) {
			PyErr_SetString(
				DokiError,
				"target_list must be a list of qubit ids (unsigned integers)");
			return NULL;
		}
		if ((num_controls > 0 &&
		     PySet_Contains(control_set, raw_val)) ||
		    (num_anticontrols > 0 &&
		     PySet_Contains(acontrol_set, raw_val))) {
			PyErr_SetString(
				DokiError,
				"A target cannot also be a control or an anticontrol");
			return NULL;
		}
		targets[i] = PyLong_AsLong(raw_val);
		if (targets[i] >= num_qubits) {
			PyErr_SetString(DokiError, "Target qubit out of range");
			return NULL;
		}
	}

	new_state = apply_gate_fmat(state_capsule, gate_capsule, targets,
				    num_targets, controls, num_controls,
				    anticontrols, num_anticontrols);

	if (new_state == NULL) {
		switch (errno) {
		case 1:
			PyErr_SetString(DokiError,
					"[FMAPPLY] Failed to allocate matrix");
			break;
		case 5:
			PyErr_SetString(
				DokiError,
				"[FMAPPLY] Failed to allocate data struct");
			break;
		default:
			PyErr_SetString(DokiError,
					"[FMAPPLY] Unknown error code");
		}
		free(targets);
		if (num_controls > 0) {
			free(controls);
		}
		if (num_anticontrols > 0) {
			free(anticontrols);
		}
		return NULL;
	}

	return PyCapsule_New((void *)new_state, "qsimov.doki.funmatrix",
			     &doki_funmatrix_destroy);
}

static PyObject *doki_funmatrix_mem(PyObject *self, PyObject *args) {
	PyObject *fmat_capsule;
	void *raw_fmat;
	size_t size;
	int debug_enabled;

	if (!PyArg_ParseTuple(args, "Op", &fmat_capsule, &debug_enabled)) {
		PyErr_SetString(DokiError,
				"Syntax: funmatrix_mem(fmatrix, verbose)");
		return NULL;
	}

	raw_fmat = PyCapsule_GetPointer(fmat_capsule, "qsimov.doki.funmatrix");
	if (raw_fmat == NULL) {
		PyErr_SetString(DokiError, "NULL pointer to FMatrix");
		return NULL;
	}

	size = FM_mem_size((struct FMatrix *)raw_fmat);

	return PyLong_FromSize_t(size);
}
