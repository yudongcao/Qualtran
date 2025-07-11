from typing import Dict, Set, List
import attrs
from functools import cached_property
from qualtran import Bloq, Signature, QInt, Register, QBit, BloqBuilder, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Toffoli
from qualtran.drawing import show_bloq
#from qualtran.simulation.classical_sim import add_ints, ClassicalValT
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
import quimb.tensor as qtn
import numpy as np



class MAJ(Bloq):

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('c', QBit()), Register('b', QBit()), Register('a', QBit())])

    def build_composite_bloq(
        self, bb: BloqBuilder, c: Soquet, b: Soquet, a: Soquet
    ) -> Dict[str, SoquetT]:
        a, b = bb.add(CNOT(), ctrl=a, target=b)
        a, c = bb.add(CNOT(), ctrl=a, target=c)
        (b, c), a = bb.add(Toffoli(), ctrl=[b, c], target=a)
        return {"c": c, "b": b, "a": a}


class UMA(Bloq):

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('c', QBit()), Register('s', QBit()), Register('a', QBit())])

    def build_composite_bloq(
        self, bb: BloqBuilder, c: Soquet, s: Soquet, a: Soquet
    ) -> Dict[str, SoquetT]:
        (c, s), a = bb.add(Toffoli(), ctrl=[c, s], target=a)
        a, c = bb.add(CNOT(), ctrl=a, target=c)
        c, s = bb.add(CNOT(), ctrl=c, target=s)
        return {"c": c, "s": s, "a": a}


@attrs.frozen
class CuccaroADD(Bloq):

    bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('a', QInt(self.bitsize)), Register('b', QInt(self.bitsize + 1))])

    def build_composite_bloq(
        self, bb: 'BloqBuilder', a: 'Soquet', b: 'Soquet'
    ) -> Dict[str, 'SoquetT']:

        n = self.bitsize
        a = bb.split(a)[::-1]
        b = bb.split(b)[::-1]
        c = bb.allocate(dtype=QBit())
        for i in range(n):
            if i == 0:
                c, b[i], a[i] = bb.add(MAJ(), c=c, b=b[i], a=a[i])
            else:
                a[i - 1], b[i], a[i] = bb.add(MAJ(), c=a[i - 1], b=b[i], a=a[i])

        a[n - 1], b[n] = bb.add(CNOT(), ctrl=a[n - 1], target=b[n])

        for i in range(n - 1, -1, -1):
            if i == 0:
                c, b[i], a[i] = bb.add(UMA(), c=c, s=b[i], a=a[i])
            else:
                a[i - 1], b[i], a[i] = bb.add(UMA(), c=a[i - 1], s=b[i], a=a[i])

        bb.free(c)
        a = bb.join(a[::-1])
        b = bb.join(b[::-1])
        return {'a': a, 'b': b}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(MAJ(), self.bitsize), (UMA(), self.bitsize), (CNOT(), 1)}

    # def on_classical_vals(
    #     self, a: 'ClassicalValT', b: 'ClassicalValT'
    # ) -> Dict[str, 'ClassicalValT']:
    #     return {'a': a, 'b': add_ints(int(a), int(b), num_bits=int(self.bitsize), is_signed=True)}

@attrs.frozen
class MyCuccaroADD(Bloq):
    
    cuccaro : CuccaroADD
    bitsize : int
    
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('a', QInt(self.bitsize)), Register('b', QInt(self.bitsize + 1))])

    def my_tensors(self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT'])-> List['qtn.Tensor']:
        
        matrix = np.array(self.cuccaro.tensor_contract(),dtype=np.complex128)
        tensor = matrix.reshape((2,2,2,2))
        outgoing_inds = [
            (outgoing['a'], [1,0]),
            (outgoing['b'], [1,0,1])
        ]
        incoming_inds = [
            (incoming['a'], [1,0]),
            (incoming['b'], [1,1,0])
        ]
        
        return [qtn.Tensor(
            data=tensor, 
            inds=outgoing_inds + incoming_inds
        )]
        