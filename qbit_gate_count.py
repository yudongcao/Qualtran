## Script for computing the qubit and T gate counts for a recursive multiplication circuit based on Toom-Cook algorithm.

import matplotlib.pyplot as plt
from math import log

POLICY_OPTIONS = ['Default', 'Karatsuba', 'Toom-3']
# Set policy for determining the parameter k at each level of recursion.
# - Default: padding the number of qubits until it is divisible by either 2 or 3.
# - Karatsuba: always use k=2. If n is odd, pad it to the next even number.
# - Toom-3: always use k=3. If n is not divisible by 3, pad it to the next multiple of 3.

def implement_policy(n, policy):
    """
    Implement the policy for determining the parameter k.
    :param n: Number of bits in the input numbers (assumed to be equal).
    :param policy: policy for determining the parameter k.
    :return: num_qubits, k - updated number of input bits and k value based on the policy.
    """

    num_qubits = int(n)

    if policy == 'Default':
        # Default policy: k is either 2 or 3, depending on n
        if n % 3 == 0:
            k = 3
        elif n % 2 == 0:
            k = 2
        else:
            num_qubits += 1  # Pad n to make it even
            k = 2
    elif policy == 'Karatsuba':
        # Karatsuba policy: always use k=2
        if n % 2 != 0:
            num_qubits += 1
        k = 2
    elif policy == 'Toom-3':
        # Toom3 policy: always use k=3
        if n % 3 != 0:
            num_qubits += (3 - n % 3)
        k = 3

    return num_qubits, k

def qubit_count(n, policy='Default', method='Toom-Cook'):
    """
    Calculate the number of qubits required for a Toom-Cook multiplication circuit.
    :param n: Number of bits in the input numbers (assumed to be equal).
    :param policy: policy for determining the parameter k, which is the number of interpolation points in the Toom-Cook algorithm.
    :param method: method for multiplication, either 'Trivial' or 'Toom-Cook'.
    :return: Number of qubits required.
    """

    if method == 'Trivial':
        return n * 2  # For trivial multiplication, we need 2 qubits per bit (one for each input bit).
    
    elif method == 'Toom-Cook':

        num_qubits, k = implement_policy(n, policy)

        if num_qubits > 6: # if this is not base case
            if k == 2:
                return qubit_count(num_qubits/2, policy) + qubit_count(num_qubits/2+1, policy)
            if k == 3:
                return qubit_count(num_qubits/3, policy) + 2*qubit_count(num_qubits/3+2, policy)
        else: # base cases
            if num_qubits in [2,3,4,6]:
                return num_qubits * 2

def T_gate_count(n, policy='Default', epsilon=1e-6, method='Toom-Cook'):
    """
    Calculate the number of T gates required for a Toom-Cook multiplication circuit.
    :param n: Number of bits in the input numbers (assumed to be equal).
    :param policy: policy for determining the parameter k.
    :param epsilon: error tolerance parameter for compiling CRz gates to Clifford+T gates.
    :param method: method for multiplication, either 'Trivial' or 'Toom-Cook'.
    :return: Number of T gates required.
    """
    
    CRz_cost = 3.45 * log(1/epsilon)/log(2) # approximate cost of a CRz gate in T gates

    if method == 'Trivial':
        # For trivial multiplication, we have n^2 CRz gates.
        return n**2 * CRz_cost

    elif method == 'Toom-Cook':

        num_qubits, k = implement_policy(n, policy)
        
        if num_qubits > 6:  # if this is not base case
            if k == 2:
                return 2*T_gate_count(num_qubits/2, policy) + T_gate_count(num_qubits/2+1, policy) + 16*(num_qubits-1)
            if k == 3:
                return 2*T_gate_count(num_qubits/3, policy) + 3*T_gate_count(num_qubits/3+2, policy) + 352*num_qubits/3 + 592
        else:  # base cases
            if num_qubits == 2:
                return 4 * CRz_cost
            elif num_qubits == 3:
                return 9 * CRz_cost
            elif num_qubits == 4:
                return 4 * T_gate_count(2)
            elif num_qubits == 6:
                return 4 * T_gate_count(3)

if __name__ == "__main__":
    # Example usage
    n = 8  # Number of bits in the input numbers
    qubits_needed = qubit_count(n)
    print(f"Number of qubits needed for {n}-bit Toom-Cook multiplication: {qubits_needed}")

    # Plotting the qubit count for different sizes
    sizes = list(range(2, 4096))
    plt.figure(figsize=(10,6))

    for policy in POLICY_OPTIONS:
        qubit_counts = [qubit_count(size, policy=policy) for size in sizes]
        # plot with thicker lines for visibility
        plt.plot(
            sizes,
            qubit_counts,
            label=policy,       # legend label
            linewidth=2.5       # make the line bolder
        )
    qubit_counts_trivial = [qubit_count(size, policy='Default', method='Trivial') for size in sizes]
    plt.plot(
        sizes,
        qubit_counts_trivial,
        label='Trivial Multiplication',  # legend label
        linestyle='--',  # dashed line for trivial method
        linewidth=2.5    # make the line bolder
    )

    plt.title('Qubit Count for Toom-Cook Multiplication', fontsize=14)
    plt.xlabel('Number of Bits (n)', fontsize=12)
    plt.ylabel('Qubit Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Policy choice', fontsize=10, title_fontsize=11)
    plt.tight_layout()

    # Plotting the T gate count for different sizes
    plt.figure(figsize=(10,6))
    for policy in POLICY_OPTIONS:
        T_gate_counts = [T_gate_count(size, policy=policy) for size in sizes]
        # plot with thicker lines for visibility
        plt.plot(
            sizes,
            T_gate_counts,
            label=policy,       # legend label
            linewidth=2.5       # make the line bolder
        )
    T_gate_counts_trivial = [T_gate_count(size, policy='Default', method='Trivial') for size in sizes]
    plt.plot(
        sizes,
        T_gate_counts_trivial,
        label='Trivial Multiplication',  # legend label
        linestyle='--',  # dashed line for trivial method
        linewidth=2.5    # make the line bolder
    )

    plt.title('T Gate Count for Toom-Cook Multiplication', fontsize=14)
    plt.xlabel('Number of Bits (n)', fontsize=12)
    plt.ylabel('T Gate Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Method', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.show()