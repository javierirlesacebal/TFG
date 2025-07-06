import doki
import numpy as np
from numpy.random import SeedSequence
import gc
from mpi4py import MPI
import scipy.sparse as sparse
import time
import sys
import ctypes
import psutil

# -------------------------------- TESTS DE CREACION DE REGISTROS  --------------------------------
def irles_check_range(min_qubits, max_qubits, with_data=False, with_lists=False):
    for nq in range(min_qubits, max_qubits + 1):
        irles_check_generation(nq, with_data=with_data, with_lists=with_lists)
        time.sleep(0.1)

def irles_check_generation(num_qubits, with_data=False, with_lists=False):
    registry_data = gen_reg_complex(num_qubits, with_data=with_data)
    if with_data:
        aux = registry_data.reshape(registry_data.shape[0])
        if with_lists:
            r_doki = doki.master_registry_new_data(num_qubits, list(aux), 0)
        else:
            r_doki = doki.master_registry_new_data(num_qubits, aux, 0)
    else:
        r_doki = doki.master_registry_new(num_qubits, 0)
    if not all(doki_master_to_np(r_doki, num_qubits) == registry_data):
        error("Error comparing results of two qubit gate", fatal=True)

def irles_reg_creation_test(min_qubits, max_qubits):
    print("\tEmpty initialization tests...")
    irles_check_range(min_qubits, max_qubits, with_data=False, with_lists=False)
    print("\tRegistry list initialization tests...")
    irles_check_range(min_qubits, max_qubits, with_data=True, with_lists=True)
    print("\tRegistry numpy initialization tests...")
    irles_check_range(min_qubits, max_qubits, with_data=True, with_lists=False)
    print(f"\tPEACE AND TRANQUILITY")

# --------------------------------  ONE GATE TESTS --------------------------------
def irles_one_gate_tests(min_qubits, max_qubits, prng):
    for nq in range(min_qubits, max_qubits + 1):
        irles_test_gates_static(nq, prng)
    for _ in range(15):
        print(f"\tPEACE AND TRANQUILITY")

def irles_test_gates_static(n_qubits, _prng):
    rtol, atol = 0, 1e-13
    valores_registro = gen_reg_complex(n_qubits)
    _registro = doki.master_registry_new(n_qubits, 0)
    for iteration in range(n_qubits):
        valores_aux = valores_registro
        registro_aux = _registro
        angulos, invertir = np.pi * (_prng.random(3) * 2 - 1), _prng.choice(a=[False, True])
        valores_registro, _registro = apply_master_state_gate(n_qubits, valores_aux, registro_aux, U_sparse(*angulos, invertir), U_doki(*angulos, invertir), iteration)
        _resultado = doki_master_to_np(_registro, n_qubits)
        if not np.allclose(_resultado, valores_registro, rtol=rtol, atol=atol):
            print("iteration:", iteration)
            print("angulos:", angulos)
            print("invertir:", invertir)
            print("valores_aux:\n", valores_aux)
            print("registro_aux:\n", doki_master_to_np(registro_aux, n_qubits))
            print("np registro:", valores_registro)
            print("doki regristro:", _resultado)
            print("comp:", np.allclose(_resultado, valores_registro, rtol=rtol, atol=atol))
            error("Error applying gate", fatal=True)
        del valores_aux
        del registro_aux

#  -------------------------------- TEST DE PROBABILIDAD  --------------------------------

def irles_probability_test(min_qubits, max_qubits):
    rtol, atol = 0, 1e-13
    for nq in range(min_qubits, max_qubits + 1):
        irles_test_probability(nq, rtol, atol)
    print(f"\tPEACE AND TRANQUILITY")

def irles_test_probability(nq, rtol=0, atol=1e-13):
    step = 2 * np.pi / nq                                                                   # Valido
    gates = [Ry_doki(step * i) for i in range(nq)]                                          # Valido
    reg = doki.master_registry_new(nq, False)                                               # Actualizado
    for i in range(nq):                                                                     # Valido
        aux = doki.master_registry_apply(reg, gates[i], [i], None, None, 0)                 # Actualizado
        del reg                                                                             # Actualizado
        reg = aux                                                                           # Valido
    for i in range(nq):                                                                     # Valido
        np_old = doki_master_to_np(reg, nq)                                                 # Actualizado
        odds = doki.master_registry_prob(reg, i)                                            # Actualizado
        exodds = np.sin((step * i) / 2)**2                                                  # Valido
        if not np.allclose(odds, exodds, rtol=rtol, atol=atol):                             # Valido
            print(f"\tDoki Odds: P(M({i})=1) = {odds}")
            print(f"\tTheorical Odds: P(M({i})=1) = {exodds}")
            np_reg = doki_master_to_np(reg, nq)
            print(f"\tr_doki: {np_reg}")
            print(f"\tequals old: {np.all(np_reg == np_old)}")
            print(f"\tsum: {np.sum(np.abs(np_reg)**2)}")                                    # Actualizado
            raise Exception(f"Failed probability check")
    del reg
    print(f"\tPEACE AND TRANQUILITY")


#  -------------------------------- TEST DE MEDIDA  --------------------------------

def irles_measure_tests(min_qubits, max_qubits, iterations, prng, rtol = 0, atol = 1e-12):
    print("\tSuperposition tests...")
    for nq in range(min_qubits, max_qubits + 1):
        pass
        irles_check_measure_superposition(nq, rtol, atol, iterations, prng)
    gc.collect()
    print("\tClassic tests...")
    for nq in range(min_qubits, max_qubits + 1):
        irles_check_measure_classic(nq, rtol, atol, iterations, prng)
    gc.collect()
    for _ in range(20):
        print('TODO CORRECTO Y YO QUE ME ALEGRO')


def irles_check_measure_superposition(num_qubits, rtol, atol, iterations, prng, bounds=None):
    # Test measurement with specified number of qubits and Hadamards
    if bounds is None:                                                                                          # Actualizado
        bounds = [.3, .4, .4, .45, .55, .6, .6, .7]                                                             # Actualizado

    h_e_doki, h_e_sp = get_H_e(num_qubits)                                                                      # Valido
    r_doki, r_np = irles_check_build(num_qubits, h_e_doki, h_e_sp, rtol, atol)
    irles_check_nothing(num_qubits, r_doki, rtol, atol)


    if num_qubits > 1:
        print("\t\tTesting mask = half")
        irles_check_half(num_qubits, r_doki, rtol, atol, iterations, bounds, prng, r_np)
    print("\t\tTesting mask = max")

    irles_check_everything(num_qubits, r_doki, iterations, bounds, prng)
    del r_doki
    del r_np

def irles_check_build(num_qubits, h_e_doki, h_e_sp, rtol, atol):
    """Test registry after gate application."""
    r_np = gen_reg_complex(num_qubits, with_data=False)
    r_doki = doki.master_registry_new(num_qubits, 0)
    for i in range(num_qubits):
        if h_e_sp[i] is not None and h_e_doki[i] is not None:
            aux_r_n, aux_r_d = irles_apply_gate(num_qubits, r_np, r_doki, h_e_sp[i], h_e_doki[i], i)
            del r_doki
            del r_np
            r_doki = aux_r_d
            r_np = aux_r_n

    doki_result = doki_master_to_np(r_doki, num_qubits)
    if not np.allclose(doki_result, r_np, rtol=rtol, atol=atol):
        print(f'Resultado doki: {doki_result}')
        print(f'Resultado NumPy: {r_np}')
        error("Error applying gates", fatal=True)
    return r_doki, r_np

def irles_check_nothing(num_qubits, r_doki, rtol, atol):
    """Test measurement with mask 0 for specified number of qubits."""
    aux_r_d, m = doki.master_registry_measure(r_doki, 0, [], 0)
    aux_r_d_np = doki_master_to_np(r_doki, num_qubits)
    m_np = doki_master_to_np(aux_r_d, num_qubits)
    if (not np.allclose(aux_r_d_np, m_np, rtol=rtol, atol=atol)) or len(m) != num_qubits or not all(x is None for x in m):
        error("Error with mask 0", fatal=True)
    del m

def irles_check_half(num_qubits, r_doki, rtol, atol, iterations, bounds, prng, r_np, classic=None):
    """Test measuring half qubits."""
    ids = [i for i in range(num_qubits)]
    mess = np.zeros((iterations, len(ids) // 2), dtype=int)
    nots = []
    np.random.seed(int(prng.integers(0, 2**32)))
    original = doki_master_to_np(r_doki, num_qubits)
    for i in range(iterations):
        not_ids = np.random.choice(ids, size=len(ids)//2, replace=False)
        nots.append(not_ids)
        yes_ids = np.setdiff1d(ids, not_ids)
        yes_ids.sort()
        mask = int(np.array([2**_id for _id in not_ids]).sum())
        aux_r_d, mes = doki.master_registry_measure(r_doki, mask, prng.random(len(not_ids)).tolist(), 0)
        aux_r_np = extract_reduced_state(r_np, mes)
        # noinspection PyTypeChecker
        result = irles_compare_no_phase(len(yes_ids), aux_r_d, aux_r_np, rtol, atol)
        if not result or len(mes) != num_qubits:
            print("Numpy:", aux_r_np)
            # noinspection PyTypeChecker
            print("Doki collapsed:", doki_master_to_np(aux_r_d, len(yes_ids)))
            print("ORIGINAL:")
            for element in original:
                print(f"\t{element}")
            print(f'yes_ids = {yes_ids}')
            print(f'not_ids = {not_ids}')
            print(f'mes = {mes}')


            error("Error measuring half qubits. Mask:", mask, fatal=True)

        mes = mes[::-1]
        for j in range(len(not_ids)):
            mess[i, j] = int(mes[not_ids[j]])
        doki.master_registry_del(aux_r_d)
        gc.collect()

    gc.collect()

    if classic is None:
        check_statistics(mess, iterations, bounds)
        return None

    expected = [[classic[num_qubits - _id - 1] for _id in nots[i]] for i in range(iterations)]
    if not all(np.all(mess[i] == expected[i]) for i in range(iterations)):
        for i in range(iterations):
            if np.all(mess[i] == expected[i]):
                continue
            print(classic)
            print(nots[i])
            print(mess[i])
            print(expected[i])
        error("Value differs from expected", fatal=True)

def irles_check_everything(num_qubits, r_doki, iterations, bounds, prng, classic=None):
    """Test measurement with max mask for specified number of qubits."""

    aux_r_d, m = doki.master_registry_measure(r_doki, 2**num_qubits - 1, prng.random(num_qubits).tolist(), 0)
    doki_errored = False
    try:
        doki.registry_get(aux_r_d, 0, 0, 0)
    except Exception as e:
        if type(e).__module__ + "." + type(e).__name__ == "qsimov.doki.error":
            doki_errored = True
    if not doki_errored or len(m) != num_qubits or not all(x is True or x is False for x in m):
        error("Error measuring all qubits", fatal=True)

    mess = np.zeros((iterations, num_qubits), dtype=int)
    for i in range(iterations):
        reg, mes = doki.master_registry_measure(r_doki, 2**num_qubits - 1, prng.random(num_qubits).tolist(), 0)
        if reg is not None:
            doki.master_registry_del(reg)
        for j in range(num_qubits):
            mess[i, j] = int(mes[j])

    if classic is None:
        check_statistics(mess, iterations, bounds)
    else:
        if not all(np.all(mes == classic) for mes in mess):
            for i in range(iterations):
                if not np.all(mess[i] == classic):
                    print(classic)
                    print(mess[i])
            error("Value differs from expected", fatal=True)

    del mess

def irles_check_measure_classic(num_qubits, rtol, atol, iterations, prng):
    """Test measurement with specified number of qubits and X gates."""
    raw_x = [[0, 1], [1, 0]]
    x_sp = sparse.csr_matrix(raw_x)
    x_d = doki.gate_new(1, raw_x, 0)
    values = prng.choice(a=[0, 1], size=num_qubits)
    x_d_list = [x_d if value else None for value in values]
    x_sp_list = [x_sp if value else None for value in values]
    r_doki, r_np = irles_check_build(num_qubits, x_d_list, x_sp_list, rtol, atol)

    print("\t\tTesting mask = max")
    irles_check_everything(num_qubits, r_doki, iterations, None, prng, values[::-1])

    if (num_qubits > 1):
        print("\t\tTesting mask = half")
        irles_check_half(num_qubits, r_doki, rtol, atol, iterations, None, prng, r_np, values[::-1])
    del r_doki
    del r_np

def irles_compare_no_phase(num_qubits, r_doki, r_np, rtol, atol):
    """Compare Doki and Numpy registry ignoring hidden phase."""
    r_d_np = doki_master_to_np(r_doki, num_qubits)
    darg = np.angle(r_d_np[0, 0])
    r_d_np = np.power(np.e, darg * -1j) * r_d_np
    narg = np.angle(r_np[0, 0])
    r_np = np.power(np.e, narg * -1j) * r_np
    return np.allclose(r_d_np, r_np, rtol=rtol, atol=atol)

def irles_apply_gate(nq, r_np, r_doki, g_sparse, g_doki, target):
    """Apply gate to registry (both numpy+sparse and doki)."""
    new_r_doki = doki.master_registry_apply(r_doki, g_doki, [target], None, None, 0)
    new_r_np = apply_np_gate(nq, r_np, g_sparse, target)
    return new_r_np, new_r_doki


def extract_reduced_state(full_state, results):
    measured = list(results)
    measured.reverse()
    # Índices de qubits no medidos (se conservan)
    free_qubits = [i for i, val in enumerate(measured) if val is None]
    n_free = len(free_qubits)
    reduced_state = np.zeros((2 ** n_free, 1), dtype=complex)

    for i in range(2 ** len(measured)):
        # Verificar que coincide con los qubits medidos
        valid = True
        for q, val in enumerate(measured):
            if val is not None:
                if ((i >> q) & 1) != int(val):
                    valid = False
                    break
        if not valid:
            continue

        # Construir índice reducido solo con qubits no medidos
        reduced_index = 0
        for j, q in enumerate(free_qubits):
            if (i >> q) & 1:
                reduced_index |= (1 << j)
        reduced_state[reduced_index] += full_state[i]
    # Normalizar
    norm = np.linalg.norm(reduced_state)
    if norm != 0:
        reduced_state /= norm
    return reduced_state


def H_e_np(quantity):
    # Original, no he modificado nada
    angle = (2 * np.pi) / quantity
    sqrt2_2 = np.sqrt(2) / 2
    h = np.array([[sqrt2_2, sqrt2_2], [sqrt2_2, -sqrt2_2]])
    return [np.dot(RZ_np(angle, False), h) for _ in range(quantity)]


def get_H_e(quantity):
    # Original, no he modificado nada
    h_es = H_e_np(quantity)
    h_e_doki = [doki.gate_new(1, h_e.tolist(), False) for h_e in h_es]
    h_e_sp = [sparse.csr_matrix(h_e) for h_e in h_es]
    return h_e_doki, h_e_sp

def RZ_np(angle, invert):
    """Return numpy array with rotation gate around Z axis."""
    gate = np.zeros(4, dtype=complex).reshape(2, 2)
    if not invert:
        gate[0, 0] = np.cos(-angle/2) + np.sin(-angle/2) * 1j
        gate[1, 1] = np.cos(angle/2) + np.sin(angle/2) * 1j
    else:
        gate[0, 0] = np.cos(-angle/2) - np.sin(-angle/2) * 1j
        gate[1, 1] = np.cos(angle/2) - np.sin(angle/2) * 1j
    return gate

def check_statistics(mess, iterations, bounds):
    """Test if the measure chances are within the expected range."""
    nq = mess.shape[1]
    total = mess.sum() / (iterations * nq)
    if total < bounds[3] or total > bounds[4]:
        print("\t[WARNING] Total chance", total * 100)
        print("\t\tTry to repeat the test with more iterations")
        if total < bounds[2] or total > bounds[5]:
            raise AssertionError("Measurement appears to be biased")
        raise AssertionError("Try to repeat the test with more iterations")

    partials = np.array([mess[:, i].sum() / iterations for i in range(nq)])
    if any(prob < bounds[1] or prob > bounds[6] for prob in partials):
        print("\t[WARNING] Chance of triggering", partials * 100)
        print("\t\tTry to repeat the test with more iterations")
        if any(x < bounds[0] or x > bounds[7] for x in partials):
            raise AssertionError("Measurement appears to be biased for some qubits")
        raise AssertionError("Try to repeat the test with more iterations")


#  -------------------------------- TIMED TESTS  --------------------------------
def irles_timed_tests(min_qubits, max_qubits, prng):
    x_np = np.array([[0, 1], [1, 0]])
    x_doki = doki.gate_new(1, x_np.tolist(), 0)

    sqrt2_2 = np.sqrt(2) / 2
    h_np = np.array([[sqrt2_2, sqrt2_2], [sqrt2_2, -sqrt2_2]])
    h_doki = doki.gate_new(1, h_np.tolist(), 0)

    times = np.empty(max_qubits + 1 - min_qubits, dtype=float)
    for num_qubits in range(min_qubits, max_qubits + 1):
        print(f"\tChecking time needed with {num_qubits} qubits...")
        toggle = True
        controls = {0}
        anticontrols = set()
        a = time.time()
        r_doki = doki.master_registry_new(num_qubits, 0)
        r_doki = doki.master_registry_apply(r_doki, h_doki, [0], None, None, 0)
        diff = time.time() - a

        for i in range(1, num_qubits):
            a = time.time()
            r_doki = doki.master_registry_apply(r_doki, x_doki, [i], controls, anticontrols, 0)
            diff += (time.time() - a)
            if toggle:
                anticontrols.add(i)
            else:
                controls.add(i)
            toggle = not toggle
        a = time.time()
        r_doki, _ = doki.master_registry_measure(r_doki, (1 << num_qubits - 1), [prng.random() for i in range(num_qubits)], 0)


        diff += (time.time() - a)
        times[num_qubits-min_qubits] = diff
        print(f"\t\t{diff} s")
    print("\tTimes:", str(times).replace("\n", "\n\t       "))
    print("\tTotal:", times.sum(), "s")


#  -------------------------------- JOIN REGISTRY TESTS  --------------------------------
def irles_join_reg_test_all(min_qubits, max_qubits, prng, rtol = 0, atol = 1e-13):
    """ DO all tests."""
    a = time.time()
    for nq in range(min_qubits, max_qubits):
        irles_join_reg_test(nq, prng, rtol = rtol, atol = atol)
    b = time.time()
    time.sleep(1)
    print(f"\tPEACE AND TRANQUILITY: {b - a} s")

def irles_join_reg_test(nq, prng, rtol = 0, atol = 1e-13):
    """Test joining nq registries of one qubit."""
    gates = [U_doki(*(np.pi * (prng.random(3) * 2 - 1)), prng.choice(a=[False, True])) for _ in range(nq)]
    regs  = [doki.master_registry_new(1, 0) for _ in range(nq)]
    r2s = [doki.master_registry_apply(regs[i], gates[i], [0], None, None, 0) for i in range(nq)]
    exreg = doki.master_registry_new(nq, 0)

    for i in range(nq):
        if regs[nq - i - 1] is not None:
            doki.master_registry_del(regs[nq - i - 1])
        aux = doki.master_registry_apply(exreg, gates[i], [nq - i - 1], None, None, 0)
        if exreg is not None:
            doki.master_registry_del(exreg)
        exreg = aux
    first = True
    res = r2s[0]
    for reg in r2s[1:]:
        aux = doki.master_registry_join(res, reg, 0)
        if not first:
            del res
        first = False
        res = aux
    if not np.allclose(doki_master_to_np(res, nq, False), doki_master_to_np(exreg, nq, False), rtol=rtol, atol=atol):
        raise AssertionError("Failed right join comparison")
    if not first:
        del res
    res = r2s[-1]
    first = True

    for reg in r2s[nq-2::-1]:
        aux = doki.master_registry_join(reg, res, 0)
        if not first:
            del res
        first = False
        res = aux

    if not np.allclose(doki_master_to_np(res, nq, False), doki_master_to_np(exreg, nq, False), rtol=rtol, atol=atol):
        error("Failed left join comparison", fatal=True)
    for i in range(nq):
        del r2s[nq - i - 1]
    if not first:
        del res
    del exreg
    print(f'Registry join working for {nq} Qubits')

#  -------------------------------- CANONICAL FORM TESTS  --------------------------------
def irles_canonical_form_tests(min_qubits, max_qubits, prng, rtol = 0, atol = 1e-13):
    """ DO all tests."""
    a = time.time()
    for nq in range(min_qubits, max_qubits + 1):
        irles_test_canonical_apply(nq, prng, rtol=rtol, atol=atol)
        if nq > 1:
            irles_test_canonical_join_mes(nq, prng, rtol=rtol, atol=atol)
    b = time.time()
    for _ in range(25):
        print(f'JESUCRISTO DIO SU VIDA POR NOSOTROS, NOSOTROS SOLO TE PEDIMOS TU DINERO')


def irles_test_canonical_apply(nq, prng, rtol = 0, atol = 1e-13):
    """Test canonical get with nq qubit registries after gate apply."""
    gates = [phase_doki(np.pi * (prng.random() * 2 - 1)) for _ in range(nq)]
    reg = doki.master_registry_new(nq, 0)
    npreg = doki_master_to_np(reg, nq, canonical=False)
    npreg = np.exp(-1j*np.angle(npreg[0, 0])) * npreg
    if not np.allclose(doki_master_to_np(reg, nq, canonical=True), npreg, rtol=rtol, atol=atol):
        raise AssertionError("Failed canonical get on clean state")

    for i in range(nq):
        aux = doki.master_registry_apply(reg, gates[i], [i], None, None, 0)
        doki.master_registry_del(reg)
        reg = aux
        rawnpreg = doki_master_to_np(reg, nq, canonical=False)
        npreg = np.exp(-1j*np.angle(rawnpreg[0, 0])) * rawnpreg
        if not np.allclose(doki_master_to_np(reg, nq, canonical=True), npreg, rtol=rtol, atol=atol):
            raise AssertionError("Failed canonical get after operating")


    for i in range(nq - 1):
        aux, _ = doki.master_registry_measure(reg, 1, [prng.random()], 0)
        doki.master_registry_del(reg)
        reg = aux
        npreg = doki_master_to_np(reg, nq-i-1, canonical=False)
        npreg = np.exp(-1j*np.angle(npreg[0, 0])) * npreg
        if not np.allclose(doki_master_to_np(reg, nq - i - 1, canonical=True), npreg, rtol=rtol, atol=atol):
            error("Failed canonical get on measured state", fatal=True)
    if reg is not None:
        doki.master_registry_del(reg)


def irles_test_canonical_join_mes(nq, prng, rtol = 0, atol = 1e-13):
    """Test canonical get with nq qubit registries after join and measure."""

    gates = [phase_doki(np.pi * (prng.random() * 2 - 1)) for _ in range(nq)]
    rawregs = [doki.master_registry_new(1, 0) for _ in range(nq)]
    regs = [doki.master_registry_apply(rawregs[i], gates[i], [0], None, None, 0) for i in range(nq)]

    joined = regs[0]
    for i in range(nq):
        if rawregs[nq-i-1] is not None:
            doki.master_registry_del(rawregs[nq-i-1])

    for i in range(1, nq):
        aux = doki.master_registry_join(joined, regs[i], 0)
        if i > 1 and joined is not None:
            doki.master_registry_del(joined)
        joined = aux
        npreg = doki_master_to_np(joined, i + 1, canonical=False)
        npreg = np.exp(-1j*np.angle(npreg[0, 0])) * npreg
        if not np.allclose(doki_master_to_np(joined, i + 1, canonical=True), npreg, rtol=rtol, atol=atol):
            error("Failed canonical get on joined state", fatal=True)

    for i in range(nq):
        if regs[nq-i-1] is not None:
            doki.master_registry_del(regs[nq-i-1])

    for i in range(nq - 1):
        aux, _ = doki.master_registry_measure(joined, 1, [prng.random()], 0)
        if joined is not None:
            doki.master_registry_del(joined)
        joined = aux
        npreg = doki_master_to_np(joined, nq - i - 1, canonical=False)
        npreg = np.exp(-1j*np.angle(npreg[0, 0])) * npreg
        if not np.allclose(doki_master_to_np(joined, nq-i-1, canonical=True), npreg, rtol=rtol, atol=atol):
            error("Failed canonical get on measured state", fatal=True)
    if joined is not None:
        doki.master_registry_del(joined)

def phase_doki(angle):
    """Return a gate with no observable changes (hidden phase)."""
    npgate = np.exp(1j * angle) * np.eye(2)
    return doki.gate_new(1, npgate.tolist(), 0)

#  -------------------------------- MULTIPLE GATE TESTS --------------------------------

def irles_multiple_gate_tests(min_qubits, max_qubits, prng, rtol = 0, atol = 1e-13):
    """ DO all tests."""
    keep_executing = True
    print("\tControlled gate application tests...")
    a = time.time()
    #for nq in range(min_qubits, max_qubits + 1):
    #    irles_controlled_tests(nq, rtol, atol, prng)

    if keep_executing:
        b = time.time()
        print("\tMultiple target gate application tests...")
        c = time.time()
        for nq in range(min_qubits, max_qubits + 1):
            irles_multiple_target_tests(nq, prng, rtol=rtol, atol=atol)
        d = time.time()
        print(f"\tPEACE AND TRANQUILITY: {(b - a) + (d - c)} s")
    for _ in range(15):
        print(f'JESUCRISTO ES NUESTRO AMO Y SEÑOR, DEMOS GRACIAS AL SEÑOR, AMEN')

def irles_controlled_tests(nq, rtol, atol, prng):
    """Test application of controlled gates."""
    isControl = prng.choice(a=[False, True])
    qubitIds = [int(_id) for _id in prng.permutation(nq)]
    lastid = qubitIds[0]
    control = []
    anticontrol = []
    angles = prng.random(3)
    invert = prng.choice(a=[False, True])
    invstr = "-1" if invert else ""

    numpygate = U_sparse(*angles, invert)
    gate = U_doki(*angles, invert)
    r1_np = gen_reg_complex(nq)
    r1_doki = doki.master_registry_new(nq, 0)
    print(f'PUERTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA:')
    print(numpygate)
    r2_np, r2_doki = irles_apply_gate(nq, r1_np,  r1_doki,  numpygate, gate, lastid)

    del r1_np
    if r1_doki is not None:
        doki.master_registry_del(r1_doki)

    for _id in qubitIds[1:]:
        r1_np = r2_np
        r1_doki = r2_doki
        if isControl:
            control.append(lastid)
        else:
            anticontrol.append(lastid)
        print("\t\tid:", _id)
        print("\t\tcontrols:", control)
        print("\t\tanticontrols:", anticontrol)
        print("\t\tnq:", nq)
        print("\t\tnumpygate:", numpygate)

        r2_np = chatgpt_applyCACU_direct(numpygate, _id, control, anticontrol, nq, r1_np)

        r2_doki = doki.master_registry_apply(r1_doki, gate, [_id], set(control), set(anticontrol), 0)

        isControl = not isControl
        lastid = _id


        """ TODO: ME HE QUEDADO POR AQUI, ESTA COMPROBACION DA FALSO"""
        if not np.allclose(doki_master_to_np(r2_doki, nq), r2_np, rtol=rtol, atol=atol):
            print(f"\t\tGate: U({angles}){invstr} to qubit {lastid}")
            print(f'IMPRIMIENDO ESTADO r2_np')
            for _index, element in enumerate(r2_np):
                print(f'\t{_index}: {element}')
            print(f'IMPRIMIENDO ESTADO r2_doki')
            a = doki_master_to_np(r2_doki, nq)
            for _index, element in enumerate(a):
               print(f'\t{_index}: {element}')
            error("Error comparing results of controlled gates", fatal=True)

        del r1_np
        if r1_doki is not None:
            doki.master_registry_del(r1_doki)

def irles_multiple_target_tests(nq, prng, rtol = 0, atol = 1e-13):
    """Test multiple qubit gate."""
    angles1 = prng.random(3)
    angles2 = prng.random(3)
    invert1 = prng.choice(a=[False, True])
    invert2 = prng.choice(a=[False, True])
    numpygate = TwoU_np(*angles1, invert1, *angles2, invert2)
    sparsegate = sparse.csr_matrix(numpygate)
    dokigate = doki.gate_new(2, numpygate.tolist(), 0)
    time.sleep(0.15)
    r1_np = gen_reg_complex(nq)
    time.sleep(0.15)
    r1_doki = doki.master_registry_new(nq, 0)
    time.sleep(0.15)
    for id1 in range(nq):
        for id2 in range(nq):
            if id1 == id2:
                continue
            r2_np = apply_two_qubit_gate(sparsegate, id1, id2, nq, r1_np)
            r2_doki = doki.master_registry_apply(r1_doki, dokigate, [id1, id2], None, None, 0)
            registro_doki = doki_master_to_np(r2_doki, nq)
            if not np.allclose(registro_doki, r2_np, rtol=rtol, atol=atol):
                print(f'IMPRIMIENDO REGISTRO ORIGINAL')
                for idx, element in enumerate(r1_np):
                    print(f'{idx}: {element}')
                print(f'IMPRIMIENDO REGISTRO DE NUMPY')
                for idx, element in enumerate(r2_np):
                    print(f'{idx}: {element}')
                print(f'IMPRIMIENDO REGISTRO DE DOKI')
                for idx, element in enumerate(registro_doki):
                    print(f'{idx}: {element}')
                print(r2_np == registro_doki)
                error("Error comparing results of two qubit gate", fatal=True)
            del r2_doki

def applyCACU(gate, id, controls, anticontrols, nq, reg):
    """Apply gate with specified controls and anticontrols."""
    cset = set(controls)
    acset = set(anticontrols)
    cuac = list(cset.union(acset))
    if type(id) == list:
        extended_cuac = id + cuac
    else:
        extended_cuac = [id] + cuac
    qubitIds = [i for i in range(nq)]

    reg = negateQubits(acset, nq, reg)
    for i in range(len(extended_cuac)):
        if qubitIds[i] != extended_cuac[i]:
            indaux = qubitIds.index(extended_cuac[i])
            reg = swap_upstairs(i, indaux, nq, reg)
            qubitIds = swap_upstairs_list(i, indaux, qubitIds)
    reg = apply_np_gate(nq, reg, CU(gate, len(cuac)), 0)
    for i in range(nq):
        if qubitIds[i] != i:
            indaux = qubitIds.index(i)
            reg = swap_upstairs(i, indaux, nq, reg)
            qubitIds = swap_upstairs_list(i, indaux, qubitIds)
    reg = negateQubits(acset, nq, reg)
    return reg


def apply_two_qubit_gate(gate, target1, target2, nq, reg):
    """
    Aplica una puerta 4x4 (2 qubits) a los qubits target1 y target2 (no contiguos, cualquier orden).
    """
    reg = reg.astype(np.complex128)
    N = 1 << nq

    mask1 = 1 << target1
    mask2 = 1 << target2

    for i in range(N):
        if (i & mask1) == 0 and (i & mask2) == 0:
            idx = [
                i,
                i | mask1,
                i | mask2,
                i | mask1 | mask2
            ]

            # Determina el orden real de los qubits
            b1 = (i >> target1) & 1
            b2 = (i >> target2) & 1
            if target1 < target2:
                state_index = (b1 << 1) | b2
            else:
                state_index = (b2 << 1) | b1  # orden invertido

            vec = np.array([reg[j] for j in idx])
            res = gate @ vec
            for j, val in zip(idx, res):
                reg[j] = val

    return reg

def chatgpt_applyCACU_direct(gate, target, controls, anticontrols, nq, reg):
    """
    Aplica una puerta 1-qubit `gate` (2x2 matriz) al qubit `target` del registro `reg`,
    solo si todos los controles están en 1 y anticontroles en 0.

    Convención: qubit 0 es el LSB (bit 0 del índice).

    Parámetros:
    - gate: np.array 2x2 de números complejos
    - target: int (índice del qubit objetivo)
    - controls: lista de int (qubits que deben estar en 1)
    - anticontrols: lista de int (qubits que deben estar en 0)
    - nq: número de qubits (log2(len(reg)))
    - reg: np.array de tamaño (2**nq,) de números complejos
    """
    reg = reg.copy()
    N = 1 << nq  # 2^nq

    for i in range(N):
        if all((i >> c) & 1 for c in controls) and all(((i >> a) & 1) == 0 for a in anticontrols):
            j = i ^ (1 << target)  # índice que difiere solo en el qubit target
            if i < j:
                v0, v1 = reg[i], reg[j]
                r0 = gate[0, 0] * v0 + gate[0, 1] * v1
                r1 = gate[1, 0] * v0 + gate[1, 1] * v1
                reg[i], reg[j] = r0, r1

    return reg


#  -------------------------------- COSAS PARA QUE FUNCIONEN EL RESTO DE COSAS  --------------------------------
def sparseTwoGate(gate, raw_id1, raw_id2, nq, reg):
    """Apply a gate to two qubits that might not be next to each other."""
    if raw_id2 < raw_id1:
        id1, id2 = raw_id2, raw_id1
    else:
        id1, id2 = raw_id1, raw_id2
    if id1 < 0 or id2 >= nq:
        reg = None
    else:
        if id2 - id1 > 1:
            reg = swap_downstairs(id1, id2 - 1, nq, reg)
        if raw_id2 < raw_id1:
            reg = apply_np_gate(nq, reg, SWAP_np(), id2 - 1)
        reg = apply_np_gate(nq, reg, gate, id2 - 1)
        if raw_id2 < raw_id1:
            reg = apply_np_gate(nq, reg, SWAP_np(), id2 - 1)
        if id2 - id1 > 1:
            reg = swap_upstairs(id1, id2 - 1, nq, reg)
    return reg

def TwoU_np(angle1_1, angle1_2, angle1_3, invert1, angle2_1, angle2_2, angle2_3, invert2):
    """Return numpy two qubit gate that may entangle."""
    U1 = U_np(angle1_1, angle1_2, angle1_3, invert1)
    U2 = U_np(angle2_1, angle2_2, angle2_3, invert2)
    g1 = sparse.kron(U1, sparse.identity(2))
    g2 = np.eye(4, dtype=complex)
    g2[2, 2] = U2[0, 0]
    g2[2, 3] = U2[0, 1]
    g2[3, 2] = U2[1, 0]
    g2[3, 3] = U2[1, 1]
    g = g2.dot(g1.toarray())
    return g

def swap_downstairs_list(id1, id2, li):
    """Swap list element id1 with the next until reaches id2 (id1 > id2)."""
    for i in range(id1, id2):
        li[i], li[i+1] = li[i+1], li[i]
    return li


def swap_upstairs_list(id1, id2, li):
    """Swap list element id1 with the next until reaches id2 (id1 < id2)."""
    for i in range(id2 - 1, id1 - 1, -1):
        li[i], li[i+1] = li[i+1], li[i]
    return li

def swap_downstairs(id1, id2, nq, reg):
    """Swap qubit id1 with next qubit until reaches id2 (id1 < id2)."""
    swap = SWAP_np()
    for i in range(id1, id2):
        reg = apply_np_gate(nq, reg, swap, i)
    return reg


def swap_upstairs(id1, id2, nq, reg):
    """Swap qubit id1 with next qubit until reaches id2 (id1 > id2)."""
    swap = SWAP_np()
    for i in range(id2 - 1, id1 - 1, -1):
        reg = apply_np_gate(nq, reg, swap, i)
    return reg

def negateQubits(qubits, nq, reg):
    """Apply X gate to qubit ids specified."""
    for _id in qubits:
        reg = apply_np_gate(nq, reg, np.array([[0, 1], [1, 0]]), _id)
    return reg

def Ry_doki(angle):
    npgate = np.array([[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2),  np.cos(angle / 2)]], dtype=complex)
    return doki.gate_new(1, npgate.tolist(), 0)

def doki_master_to_np(r_doki, n_qubits, canonical=False):
    return np.transpose(np.array([doki.master_registry_get(r_doki, i, canonical, 0) for i in range(2**n_qubits)], ndmin=2))

def gen_reg(nq): # Bueno para numeros con solo parte real
    size = 1 << nq
    r = np.random.rand(size)
    r = r / np.linalg.norm(r)
    r.shape = (size, 1)
    return r

def gen_reg_complex_streaming(nq, seed=None, block_size=1 << 20):
    size = 1 << nq
    n_bytes = size * 16  # complex128 = 16 bytes
    rng = np.random.default_rng(seed)

    # Reserva única con malloc
    buf = ctypes.create_string_buffer(n_bytes)
    r = np.frombuffer(buf, dtype=np.complex128)

    # Fase 1: Generar y acumular norma²
    norm_sq = 0.0
    for i in range(0, size, block_size):
        end = min(i + block_size, size)
        real = rng.random(end - i)
        imag = rng.random(end - i)
        block = real + 1j * imag
        r[i:end] = block
        norm_sq += np.sum(np.abs(block)**2)

    # Fase 2: Normalizar
    norm = np.sqrt(norm_sq)
    r /= norm

    return r.reshape((size, 1))


def gen_reg_complex(nq, with_data=False, seed=None):
    size = 1 << nq
    if with_data:
        rng = np.random.default_rng(seed)
        real = rng.random(size)
        imag = rng.random(size)
        r = real + 1j * imag
        r = r / np.linalg.norm(r)
        r = r.reshape((size, 1))
    else:
        r = np.zeros((size, 1), dtype=complex)
        r[0, 0] = 1
    return r

def apply_master_state_gate(n_qubits, vector_estado_np, regristro, g_sparse, _puerta, target):
    resultado_doki = doki.master_registry_apply(regristro, _puerta, [target], None, None, 0)
    resultado_np = apply_np_gate(n_qubits, vector_estado_np, g_sparse, target)
    return resultado_np, resultado_doki

def apply_np_gate(n_qubits, vector_estado_np, _puerta, target):
    if _puerta is None:
        return vector_estado_np[:, :]
    if n_qubits < 2:
        return _puerta.dot(vector_estado_np)
    # Izquierda
    left = n_qubits - target - int(np.log2(_puerta.shape[0]))
    if left > 0:
        _puerta = sparse.kron(sparse.identity(2**left), _puerta)
    # Derecha
    if target > 0:
        _puerta = sparse.kron(_puerta, sparse.identity(2**target))
    return _puerta.dot(vector_estado_np)

def U_sparse(angle1, angle2, angle3, _invert):
    return sparse.csr_matrix(U_np(angle1, angle2, angle3, _invert))

def U_doki(angle1, angle2, angle3, _invert):
    return doki.gate_new(1, U_np(angle1, angle2, angle3, _invert).tolist(), 0)

def U_np(angle1, angle2, angle3, _invert):
    gate, cosan, sinan = np.zeros(4, dtype=complex).reshape(2, 2), np.cos(angle1 / 2), np.sin(angle1 / 2)
    mult = -1 if _invert else 1
    gate[0, 0] = cosan
    gate[1, 1] = cosan * np.cos(angle2 + angle3) + mult * cosan * np.sin(angle2 + angle3) * 1j
    if not _invert:
        gate[0, 1] = -sinan * np.cos(angle3) - sinan * np.sin(angle3) * 1j
        gate[1, 0] = sinan * np.cos(angle2) + sinan * np.sin(angle2) * 1j
    else:
        gate[0, 1] = sinan * np.cos(angle2) - sinan * np.sin(angle2) * 1j
        gate[1, 0] = -sinan * np.cos(angle3) + sinan * np.sin(angle3) * 1j
    return gate

def CU(gate, ncontrols):
    """Return n-controlled version of given gate."""
    nqgate = int(np.log2(gate.shape[0]))
    cu = np.eye(2**(nqgate+ncontrols), dtype=complex)
    aux = cu.shape[0] - gate.shape[0]
    for i in range(gate.shape[0]):
        for j in range(gate.shape[1]):
            cu[aux + i, aux + j] = gate[i, j]

    return sparse.csr_matrix(cu)

def SWAP_np():
    """Return numpy array with SWAP gate."""
    gate = np.zeros(4 * 4, dtype=complex)
    gate = gate.reshape(4, 4)
    for i in range(4): # MODIFICADO PELIGRO
        gate[i][i] = 1
    return gate



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mode = 4

    if mode == 4:
        seed = 69
        ss = SeedSequence(seed)
        print(f"PYTHON Using seed = {ss.entropy}")
        prng = np.random.default_rng(ss.entropy)

        """ FUNCIONAN """

        irles_join_reg_test_all(2, 9, prng)
        irles_reg_creation_test(1, 9)
        irles_one_gate_tests(1, 9, prng)
        irles_test_probability(9)
        irles_measure_tests(5, 5, 300, prng)
        irles_canonical_form_tests(2, 9, prng)
        irles_timed_tests(4, 9, prng)
        irles_multiple_gate_tests(4, 9, prng)


    elif mode == 21:
        seed = 69
        ss = SeedSequence(seed)
        print(f"PYTHON Using seed = {ss.entropy}")
        prng = np.random.default_rng(ss.entropy)

        #angles1 = prng.random(3)
        #angles2 = prng.random(3)
        #invert1 = prng.choice(a=[False, True])
        #invert2 = prng.choice(a=[False, True])
        #numpygate = TwoU_np(*angles1, invert1, *angles2, invert2)
        #dokigate = doki.gate_new(2, numpygate.tolist(), 0)

        #angles = np.pi * (prng.random(3) * 2 - 1)
        #invert = prng.choice(a=[False, True])
        #valores_puerta = U_np(*angles, invert).tolist()
        #puerta = doki.gate_new(1, valores_puerta, 0)

        try:
            runs = 4
            results = []
            for num_qubits in range(1, 28):
                valores = gen_reg_complex(num_qubits, with_data=True)
                results.append([])
                create_time = 0.0
                delete_time = 0.0

                start_global = time.perf_counter()

                for _ in range(runs):
                    t1 = time.perf_counter()
                    registro = doki.master_registry_new_data(num_qubits, valores, 0)
                    t2 = time.perf_counter()
                    doki.master_registry_del(registro)
                    t3 = time.perf_counter()

                    create_time += t2 - t1
                    delete_time += t3 - t2

                end_global = time.perf_counter()
                total_time = end_global - start_global
                results[-1].append(create_time / runs)
                results[-1].append(delete_time / runs)
                results[-1].append(total_time / runs)
                print(f'Test for {num_qubits} qubits finished')
            print(f'-----')
            for a in results:
                print(a[0])
            print(f'-----')
            for a in results:
                print(a[1])
            print(f'-----')
            for a in results:
                print(a[2])
            print(f'-----')

            #registro1 = doki.master_registry_new(numero_qubits, 0)
            #registro2 = doki.master_registry_new(numero_qubits-1, 0)
            #registro3 = doki.master_registry_new(numero_qubits-3, 0)
            #registro3 = doki.master_registry_new(numero_qubits - 3, 0)



            # valores_registro = gen_reg_complex(numero_qubits, with_data=True)
            #registro = doki.master_registry_new_data(numero_qubits, valores_registro, 0)
            #new_r_doki = doki.master_registry_apply(registro, dokigate, [0, 1], None, None, 0)
            #doki.master_registry_del(registro)
            #doki.master_registry_del(new_r_doki)
            #gc.collect()
            #time.sleep(0.1)

        except Exception as ex:
            print(f"An exception occurred {ex}")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            print("An exception occurred")
            exit(1)

        #


        #doki.master_registry_del(new_r_doki)
        gc.collect()


    elif mode == 1:
        seed = 69
        num_qubits_1 = 4
        
        ss = SeedSequence(seed)
        print(f"PYTHON using seed = {ss.entropy}")
        prng = np.random.default_rng(ss.entropy)
        registry_data1 = random_registry(num_qubits_1)
        registro = doki.master_registry_new_data(num_qubits_1, registry_data1, 0)
        angles = np.pi * (prng.random(3) * 2 - 1)
        invert = prng.choice(a=[False, True])
        valores_puerta = U_np(*angles, invert).tolist()
        puerta = doki.gate_new(1, valores_puerta, 0)
        new_r_doki = doki.master_registry_apply(registro, puerta, [3], None, None, 0)

    elif mode == 2:
        seed = 69
        num_qubits = 5
        ss = SeedSequence(seed)
        print(f"PYTHON Using seed = {ss.entropy}")
        prng = np.random.default_rng(ss.entropy)
        
        registry_data_original = gen_reg_complex(num_qubits)
        registry_data = registry_data_original.reshape(registry_data_original.shape[0])
        registry_data = list(registry_data)
        angles = np.pi * (prng.random(3) * 2 - 1)
        invert = prng.choice(a=[False, True])
        valores_puerta = U_np(*angles, invert).tolist()
        puerta = doki.gate_new(1, valores_puerta, 0)
        
        print(f'PYTHON random "registry_data" with {num_qubits} qubits')
        for index, value in enumerate(registry_data):
            print(f'\t>PYTHON registry_data[{index}]=({value})')
            
        print(f'PYTHON random "valores_puerta"')
        for index1, row in enumerate(valores_puerta):
            for index2, value in enumerate(row):
                print(f'\t>PYTHON gate_data[{index1}][{index2}]=({value})') 
        
        registro = doki.master_registry_new_data(num_qubits, registry_data, 0)
        

        resultado = doki_master_to_np(registro, num_qubits, False)
        
        if not all(resultado == registry_data_original):
            print(f'PYTHON ERROR AL COMPARAR')
        else:
            print(f'PYTHON TODO VA PERFECTO')
        print(f'RESULTADO = {resultado}')
        
        for index in range(32):
            value = doki.master_registry_get(registro, index, 0, 0)
            print(f'registro[{index}] = {value}')


        #new_r_doki = doki.master_registry_apply(registro, puerta, [2], None, None, 0)
        #del registro

        #aaa, value = doki.master_registry_measure(new_r_doki, 1, [0.000001], 0)
        aaa, value = doki.master_registry_measure(registro, 2, [0.89], 0)
        #del new_r_doki
        #aaa = doki.master_registry_measure(new_r_doki, 1, [prng.random() for i in range(num_qubits)], 0)
        
    elif mode == 3:
        seed = 69
        num_qubits_1 = 4
        num_qubits_2 = 3
        ss = SeedSequence(seed)
        print(f"PYTHON Using seed = {ss.entropy}")
        prng = np.random.default_rng(ss.entropy)
        
        registry_data1 = random_registry(num_qubits_1)
        registry_data2 = random_registry(num_qubits_2)
        
        print(f'PYTHON random registry_data1 with {num_qubits_1} qubits')
        for index, value in enumerate(registry_data1):
            print(f'PYTHON registry_data[{index}]=({value})')
        
        print(f'PYTHON random registry_data2 with {num_qubits_2} qubits')
        for index, value in enumerate(registry_data2):
            print(f'PYTHON registry_data[{index}]=({value})') 

        registro1 = doki.master_registry_new_data(num_qubits_1, registry_data1, 0)
        registro2 = doki.master_registry_new_data(num_qubits_2, registry_data2, 0)
        time.sleep(1)
        print(f'------------------------------------------------')
        registro3 = doki.master_registry_join(registro1, registro2, 0)
    elif mode == 5:
        seed = 69
        ss = SeedSequence(seed)
        print(f"PYTHON Using seed = {ss.entropy}")
        prng = np.random.default_rng(ss.entropy)

    else:
        print(f'Selected mode {mode} not asigned to any code')
    