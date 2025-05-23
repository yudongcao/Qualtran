from typing import Dict, Set, List
import attrs
from functools import cached_property
from qualtran import Bloq, Signature, QInt, Register, QBit, BloqBuilder, Soquet, SoquetT
from qualtran.bloqs.basic_gates.rotation import CZPowGate
from qualtran.drawing import show_bloq
#from qualtran.simulation.classical_sim import add_ints, ClassicalValT
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import SymbolicFloat
import quimb.tensor as qtn
import numpy as np

@attrs.frozen
class Base2(self):
    
    phi : SymbolicFloat 
    
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x0', QBit()), Register('x1', QBit()), Register('y0', QBit()), Register('y1', QBit())])
    
    def build_composite_bloq(
        self, bb: 'BloqBuilder', x0: 'Soquet', y0: 'Soquet', x1: 'Soquet', y1: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        
        phi = self.phi
        
        x0, y1 = bb.add(CZPowGate(exponent=phi), ctrl=x0, target=x1)
        x0, y1 = bb.add(CZPowGate(exponent=phi), ctrl=x0, target=y1)
        x1, y0 = bb.add(CZPowGate(exponent=phi), ctrl=y0, target=x1)
        x1, y1 = bb.add(CZPowGate(exponent=phi), ctrl=y0, target=y1)

        return {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}

@attrs.frozen
class Base3(self):
    
    phi : SymbolicFloat
    
    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x0', QBit()), Register('x1', QBit()), Register('x2', QBit()), Register('y0', QBit()), Register('y1', QBit()),  Register('y2', QBit())])
    
    def build_composite_bloq(
        self, bb: 'BloqBuilder', x0: 'Soquet', y0: 'Soquet', x1: 'Soquet', y1: 'Soquet', x2: 'Soquet', y2: 'Soquet'
    ) -> Dict[str, 'SoquetT']:
        
        phi = self.phi
        
        x0, y0 = bb.add(CZPowGate(exponent=phi), ctrl=x0, target=y0)
        x0, y1 = bb.add(CZPowGate(exponent=phi), ctrl=x0, target=y1)
        x0, y2 = bb.add(CZPowGate(exponent=phi), ctrl=x0, target=y2)
        x1, y0 = bb.add(CZPowGate(exponent=phi), ctrl=x1, target=y0)
        x1, y1 = bb.add(CZPowGate(exponent=phi), ctrl=x1, target=y1)
        x1, y2 = bb.add(CZPowGate(exponent=phi), ctrl=x1, target=y2)
        x2, y0 = bb.add(CZPowGate(exponent=phi), ctrl=x2, target=y0)
        x2, y1 = bb.add(CZPowGate(exponent=phi), ctrl=x2, target=y1)
        x2, y2 = bb.add(CZPowGate(exponent=phi), ctrl=x2, target=y2)

        return {'x0': x0, 'x1': x1, 'x2': x2, 'y0': y0, 'y1': y1, 'y2': y2}
   
        
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
        
        
        