from typing import Dict, Set, List
import attrs
from functools import cached_property
from qualtran import Bloq, Signature, QInt, Register, QBit, BloqBuilder, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CZ
from qualtran.drawing import show_bloq
#from qualtran.simulation.classical_sim import add_ints, ClassicalValT
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
import quimb.tensor as qtn
import numpy as np


class Base2(self):
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x0', QBit()), Register('x1', QBit()), Register('y0', QBit()), Register('y1', QBit())])
    
    def build_composite_bloq(
        self, bb: 'BloqBuilder', x0: 'Soquet', y0: 'Soquet', x1: 'Soquet', y1: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        
        phi = np.pi / 4
        
        x0, x1 = bb.add(CZ(), ctrl=x0, target=x1)
        x0, y1 = bb.add(CZ(), ctrl=x0, target=y1)
        y0, x1 = bb.add(CZ(), ctrl=y0, target=x1)
        y0, y1 = bb.add(CZ(), ctrl=y0, target=y1)

        return {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
    
@attrs.frozen
class PhaseProduct(Bloq):

    bitsize:int
    
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x', QInt(self.bitsize)), Register('y', QInt(self.bitsize))])
    
    def build_composite_bloq(
        self, bb: 'BloqBuilder', x: 'Soquet', y: 'Soquet', Tree: List
    ) -> Dict[str, 'SoquetT']:

        x = bb.split(x)[::-1]
        y = bb.split(y)[::-1]
        
        
        