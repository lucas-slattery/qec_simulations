import cirq
import numpy as np
from pathlib import Path
import stimcirq
from codes_constructions.syndrome_extraction_circuit_catalog import build_noiseless_syndrome_extraction_circuit, build_memory_experiment_circuit

from codes_constructions.stabilizer_code_factory import StabilizerCodeFactory

from codes_constructions.circuit_models import (
    SeqOperationConstraints,
    NoiseModel,
)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import socket
from adaptive_scheduler import SlurmExecutor
from ionq.qcircuitsim.backends.quimb.quimb_mps_simulator import QuimbMPSSimulator
from ionq.qcircuitsim.backends.cirq.cirq_qcircuit_simulator import CirqSimulatorOptions, CirqQCircuitSimulator
from ionq.qcircuitsim.qcircuit_simulator.qcircuit_simulator_options import QCircuitSimulatorOptions
import sinter
import stim
import time
from pipefunc.resources import Resources
from pipefunc import PipeFunc, Pipeline
from pipefunc.typing import Array
from itertools import chain
import json
from ast import literal_eval

def executors(
    *, allow_slurm: bool = True
) -> dict[str, ThreadPoolExecutor | ProcessPoolExecutor | SlurmExecutor]:
    """Return executors for the pipeline, only the simulations are done in parallel."""
    if socket.gethostname().startswith("obsidian") and allow_slurm:
        parallel_pool = SlurmExecutor(partition="obsidian-128",max_simultaneous_jobs=256)
    else:
        parallel_pool = ProcessPoolExecutor()
    thread_pool = ThreadPoolExecutor(max_workers=256)
    return {"result_dicts": parallel_pool, "": thread_pool}

class IonChainNoiseModel(NoiseModel):
    def __init__(self, p_CNOT: float, meas_slower_factor: float):
        self._meas_slower_factor = meas_slower_factor
        super().__init__(p_CNOT, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01 * meas_slower_factor)

    @property
    def meas_slower_factor(self) -> float:
        return self._meas_slower_factor

def build_circuit(code,constraints):
    nm = IonChainNoiseModel(p_CNOT=0.001, meas_slower_factor=30)
    circuit = build_memory_experiment_circuit(code, constraints, nm, "z", 3)

    stripped_circuit = stim.Circuit()
    for instruction in circuit:
        if instruction.name == "DEPOLARIZE1":
            continue
        if instruction.name == "DEPOLARIZE2":
            continue
        if instruction.name == "DETECTOR":
            continue
        if instruction.name == "OBSERVABLE_INCLUDE":
            continue
        if instruction.name == "M":
            stripped_circuit.append("M", instruction.targets_copy())
            continue
        stripped_circuit.append(instruction)
    cirq_circuit = stimcirq.stim_circuit_to_cirq_circuit(stripped_circuit)
    return cirq_circuit

def mps_simulate(bb_code_parameter, constraints, lazy_indexing, chi, num_repetitions):
    # Build the code
    n_data_qubits, n_logical_qubits = bb_code_parameter
    code = StabilizerCodeFactory.build_bivariate_bicycle_code(n_data_qubits, n_logical_qubits)
    cirq_circuit = build_circuit(code, constraints)
    options = CirqSimulatorOptions(QCircuitSimulatorOptions(mps_max_bond_dimension = chi, mps_track_discarded_sv_norms=True, mps_lazy_indexing=lazy_indexing))

    sim = QuimbMPSSimulator(options)
    
    try:
        time_start = time.time()

        for _ in range(num_repetitions):
            result = sim.simulate(cirq_circuit)

        time_stop = time.time()
    except ValueError:
        print("Simulation failed")
        return {}
    
    this_result = {}
    this_result["time"] = time_stop - time_start
    this_result["estimated_fidelity"] = float(result.final_state.estimation_stats()["estimated_fidelity"])
    this_result["discarded_sv_norms"] = [float(x.real) for x in result.final_state.estimation_stats()["maybe_discarded_sv_norms"]]
    this_result["norms"] = [float(x.real) for x in result.final_state.estimation_stats()["maybe_norms"]]

    num_ancillas = constraints.num_ancillas
    result_dict = {str(bb_code_parameter): {num_ancillas : {lazy_indexing: {chi: this_result}}}}
    return result_dict

def sv_simulate(bb_code_parameter, constraints, num_repetitions):
    # Build the code
    n_data_qubits, n_logical_qubits = bb_code_parameter
    code = StabilizerCodeFactory.build_bivariate_bicycle_code(n_data_qubits, n_logical_qubits)
    # Build the noiseless circuit
    cirq_circuit = build_circuit(code, constraints)

    if cirq_circuit.num_qubits() > 36:
        return {}
    else:
        options = QCircuitSimulatorOptions(num_gpus=1)

        sim = CirqQCircuitSimulator(options)
        try:
            time_start = time.time()
            for _ in range(num_repetitions):
                result = sim.run(cirq_circuit)
            time_stop = time.time()
        except ValueError:
            print("Simulation failed")
            return {}
        
        this_result = {}
        this_result["time"] = time_stop - time_start
        this_result["estimated_fidelity"] = float(result.final_state.estimation_stats()["estimated_fidelity"])
        this_result["discarded_sv_norms"] = [float(x.real) for x in result.final_state.estimation_stats()["maybe_discarded_sv_norms"]]
        this_result["norms"] = [float(x.real) for x in result.final_state.estimation_stats()["maybe_norms"]]

        result_dict = {str(bb_code_parameter) : this_result}
        return result_dict

def merge_nested_dicts(dict_list):
    """Merge a list of nested dictionaries into one dictionary."""
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key in merged_dict and isinstance(value, dict):
                merged_dict[key] = merge_nested_dicts([merged_dict[key], value])
            else:
                merged_dict[key] = value
    return merged_dict

def collate_results(result_dicts: Array[dict]):
    result_dicts = result_dicts.flatten()
    collated_results = merge_nested_dicts(result_dicts)

    return collated_results

def dump_results(collated_results,collated_result_filepath):
    with open(collated_result_filepath, 'w+') as f:
        json.dump(collated_results, f)

    return

mps_simulate_pipefuncs = [
    PipeFunc(
        mps_simulate,
        "result_dicts",
        resources=Resources(
            cpus=1,
            gpus=0,
    ),
        resources_scope="element",
        mapspec="bb_code_parameter[i], constraints[j], lazy_indexing[k], chi[l], num_repetitions ->result_dicts[i,j,k,l]",
    ),
    PipeFunc(
        collate_results,
        "collated_results",
    ),
    PipeFunc(
        dump_results,
        "end"
    ),
]

sv_simulate_pipefuncs = [
    PipeFunc(
        sv_simulate,
        "result_dicts",
        resources=Resources(
            cpus=1,
            gpus=1,
    ),
        resources_scope="element",
        mapspec="bb_code_parameter[i], constraints[j], num_repetitions ->result_dicts[i,j]",
    ),
    PipeFunc(
        collate_results,
        "collated_results",
    ),
    PipeFunc(
        dump_results,
        "end"
    ),
]

mps_pipeline = Pipeline(mps_simulate_pipefuncs)
sv_pipeline = Pipeline(sv_simulate_pipefuncs)