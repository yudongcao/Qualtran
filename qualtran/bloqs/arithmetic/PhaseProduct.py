from typing import Dict, Set, List
import attrs
from functools import cached_property
from qualtran import Bloq, Signature, QInt, Register, QBit, BloqBuilder, Soquet, SoquetT
from qualtran.bloqs.basic_gates.rotation import CZPowGate
from qualtran.drawing import show_bloq

# from qualtran.simulation.classical_sim import add_ints, ClassicalValT
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.symbolics import SymbolicFloat
import quimb.tensor as qtn
import numpy as np


@attrs.frozen
class Base2(Bloq):

    phi1: SymbolicFloat
    phi2: SymbolicFloat
    phi3: SymbolicFloat
    phi4: SymbolicFloat

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x0', QBit()),
                Register('x1', QBit()),
                Register('y0', QBit()),
                Register('y1', QBit()),
            ]
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', x0: 'Soquet', x1: 'Soquet', y0: 'Soquet', y1: 'Soquet'
    ) -> Dict[str, 'SoquetT']:

        phi1 = self.phi1
        phi2 = self.phi2
        phi3 = self.phi3
        phi4 = self.phi4

        x0, y0 = bb.add(CZPowGate(exponent=phi1), q=[x0, y0])
        x0, y1 = bb.add(CZPowGate(exponent=phi2), q=[x0, y1])
        x1, y0 = bb.add(CZPowGate(exponent=phi3), q=[x1, y0])
        x1, y1 = bb.add(CZPowGate(exponent=phi4), q=[x1, y1])

        return {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}


@attrs.frozen
class Base3(Bloq):

    phi1: SymbolicFloat
    phi2: SymbolicFloat
    phi3: SymbolicFloat
    phi4: SymbolicFloat
    phi5: SymbolicFloat
    phi6: SymbolicFloat
    phi7: SymbolicFloat
    phi8: SymbolicFloat
    phi9: SymbolicFloat

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x0', QBit()),
                Register('x1', QBit()),
                Register('x2', QBit()),
                Register('y0', QBit()),
                Register('y1', QBit()),
                Register('y2', QBit()),
            ]
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        x0: 'Soquet',
        y0: 'Soquet',
        x1: 'Soquet',
        y1: 'Soquet',
        x2: 'Soquet',
        y2: 'Soquet',
    ) -> Dict[str, 'SoquetT']:

        phi1 = self.phi1
        phi2 = self.phi2
        phi3 = self.phi3
        phi4 = self.phi4
        phi5 = self.phi5
        phi6 = self.phi6
        phi7 = self.phi7
        phi8 = self.phi8
        phi9 = self.phi9

        x0, y0 = bb.add(CZPowGate(exponent=phi1), q=[x0, y0])
        x0, y1 = bb.add(CZPowGate(exponent=phi2), q=[x0, y1])
        x0, y2 = bb.add(CZPowGate(exponent=phi3), q=[x0, y2])
        x1, y0 = bb.add(CZPowGate(exponent=phi4), q=[x1, y0])
        x1, y1 = bb.add(CZPowGate(exponent=phi5), q=[x1, y1])
        x1, y2 = bb.add(CZPowGate(exponent=phi6), q=[x1, y2])
        x2, y0 = bb.add(CZPowGate(exponent=phi7), q=[x2, y0])
        x2, y1 = bb.add(CZPowGate(exponent=phi8), q=[x2, y1])
        x2, y2 = bb.add(CZPowGate(exponent=phi9), q=[x2, y2])

        return {'x0': x0, 'x1': x1, 'x2': x2, 'y0': y0, 'y1': y1, 'y2': y2}


@attrs.frozen
class Base4(Bloq):

    phi01: SymbolicFloat
    phi02: SymbolicFloat
    phi03: SymbolicFloat
    phi04: SymbolicFloat
    phi11: SymbolicFloat
    phi12: SymbolicFloat
    phi13: SymbolicFloat
    phi14: SymbolicFloat
    phi21: SymbolicFloat
    phi22: SymbolicFloat
    phi23: SymbolicFloat
    phi24: SymbolicFloat
    phi31: SymbolicFloat
    phi32: SymbolicFloat
    phi33: SymbolicFloat
    phi34: SymbolicFloat

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x0', QBit()),
                Register('x1', QBit()),
                Register('x2', QBit()),
                Register('x3', QBit()),
                Register('y0', QBit()),
                Register('y1', QBit()),
                Register('y2', QBit()),
                Register('y3', QBit()),
            ]
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        x0: 'Soquet',
        y0: 'Soquet',
        x1: 'Soquet',
        y1: 'Soquet',
        x2: 'Soquet',
        y2: 'Soquet',
        x3: 'Soquet',
        y3: 'Soquet',
    ) -> Dict[str, 'SoquetT']:
        phi01 = self.phi01
        phi02 = self.phi02
        phi03 = self.phi03
        phi04 = self.phi04
        phi11 = self.phi11
        phi12 = self.phi12
        phi13 = self.phi13
        phi14 = self.phi14
        phi21 = self.phi21
        phi22 = self.phi22
        phi23 = self.phi23
        phi24 = self.phi24
        phi31 = self.phi31
        phi32 = self.phi32
        phi33 = self.phi33
        phi34 = self.phi34

        x0, x1, y0, y1 = bb.add(
            Base2(phi1=phi01, phi2=phi02, phi3=phi03, phi4=phi04), x0=x0, x1=x1, y0=y0, y1=y1
        )
        x2, x3, y2, y3 = bb.add(
            Base2(phi1=phi11, phi2=phi12, phi3=phi13, phi4=phi14), x0=x2, x1=x3, y0=y2, y1=y3
        )
        x0, x1, y2, y3 = bb.add(
            Base2(phi1=phi21, phi2=phi22, phi3=phi23, phi4=phi24), x0=x0, x1=x1, y0=y2, y1=y3
        )
        x2, x3, y0, y1 = bb.add(
            Base2(phi1=phi31, phi2=phi32, phi3=phi33, phi4=phi34), x0=x2, x1=x3, y0=y0, y1=y1
        )
        return {'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'y0': y0, 'y1': y1, 'y2': y2, 'y3': y3}

@attrs.frozen
class Base6(Bloq):

    phi01: SymbolicFloat
    phi02: SymbolicFloat
    phi03: SymbolicFloat
    phi04: SymbolicFloat
    phi05: SymbolicFloat
    phi06: SymbolicFloat
    phi07: SymbolicFloat
    phi08: SymbolicFloat
    phi09: SymbolicFloat
    phi11: SymbolicFloat
    phi12: SymbolicFloat
    phi13: SymbolicFloat
    phi14: SymbolicFloat
    phi15: SymbolicFloat
    phi16: SymbolicFloat
    phi17: SymbolicFloat
    phi18: SymbolicFloat
    phi19: SymbolicFloat
    phi21: SymbolicFloat
    phi22: SymbolicFloat
    phi23: SymbolicFloat
    phi24: SymbolicFloat
    phi25: SymbolicFloat
    phi26: SymbolicFloat
    phi27: SymbolicFloat
    phi28: SymbolicFloat
    phi29: SymbolicFloat
    phi31: SymbolicFloat
    phi32: SymbolicFloat
    phi33: SymbolicFloat
    phi34: SymbolicFloat
    phi35: SymbolicFloat
    phi36: SymbolicFloat
    phi37: SymbolicFloat
    phi38: SymbolicFloat
    phi39: SymbolicFloat

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x0', QBit()),
                Register('x1', QBit()),
                Register('x2', QBit()),
                Register('x3', QBit()),
                Register('x4', QBit()),
                Register('x5', QBit()),
                Register('y0', QBit()),
                Register('y1', QBit()),
                Register('y2', QBit()),
                Register('y3', QBit()),
                Register('y4', QBit()),
                Register('y5', QBit()),
            ]
        )
        
    def build_composite_bloq(self,
        bb: 'BloqBuilder',
        x0: 'Soquet',
        y0: 'Soquet',
        x1: 'Soquet',
        y1: 'Soquet',
        x2: 'Soquet',
        y2: 'Soquet',
        x3: 'Soquet',
        y3: 'Soquet',
        x4: 'Soquet',
        y4: 'Soquet',
        x5: 'Soquet',
        y5: 'Soquet',
    ) -> Dict[str, 'SoquetT']:
        phi01 = self.phi01
        phi02 = self.phi02
        phi03 = self.phi03
        phi04 = self.phi04
        phi05 = self.phi05
        phi06 = self.phi06
        phi07 = self.phi07
        phi08 = self.phi08
        phi09 = self.phi09
        phi11 = self.phi11
        phi12 = self.phi12
        phi13 = self.phi13
        phi14 = self.phi14
        phi15 = self.phi15
        phi16 = self.phi16
        phi17 = self.phi17
        phi18 = self.phi18
        phi19 = self.phi19
        phi21 = self.phi21
        phi22 = self.phi22
        phi23 = self.phi23
        phi24 = self.phi24
        phi25 = self.phi25
        phi26 = self.phi26
        phi27 = self.phi27
        phi28 = self.phi28
        phi29 = self.phi29
        phi31 = self.phi31
        phi32 = self.phi32
        phi33 = self.phi33
        phi34 = self.phi34
        phi35 = self.phi35
        phi36 = self.phi36
        phi37 = self.phi37
        phi38 = self.phi38
        phi39 = self.phi39

        x0, x1, x2, y0, y1, y2 = bb.add(
            Base3(
                phi1=phi01,
                phi2=phi02,
                phi3=phi03,
                phi4=phi04,
                phi5=phi05,
                phi6=phi06,
                phi7=phi07,
                phi8=phi08,
                phi9=phi09,
            ),
            x0=x0, x1=x1, x2=x2, y0=y0, y1=y1, y2=y2,
        )
        x3, x4, x5, y3, y4, y5 = bb.add(
            Base3(
                phi1=phi11,
                phi2=phi12,
                phi3=phi13,
                phi4=phi14,
                phi5=phi15,
                phi6=phi16,
                phi7=phi17,
                phi8=phi18,
                phi9=phi19,
            ),
            x0=x3, x1=x4, x2=x5, y0=y3, y1=y4, y2=y5,
        )
        x0, x1, x2, y3, y4, y5 = bb.add(
            Base3(
                phi1=phi21,
                phi2=phi22,
                phi3=phi23,
                phi4=phi24,
                phi5=phi25,
                phi6=phi26,
                phi7=phi27,
                phi8=phi28,
                phi9=phi29,
            ),
            x0=x0, x1=x1, x2=x2, y0=y3, y1=y4, y2=y5,
        )
        x3, x4, x5, y0, y1, y2 = bb.add(
            Base3(
                phi1=phi31,
                phi2=phi32,
                phi3=phi33,
                phi4=phi34,
                phi5=phi35,
                phi6=phi36,
                phi7=phi37,
                phi8=phi38,
                phi9=phi39,
            ),
            x0=x3, x1=x4, x2=x5, y0=y0, y1=y1, y2=y2,
        )
        return {
            'x0': x0,
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
            'x5': x5,
            'y0': y0,
            'y1': y1,
            'y2': y2,
            'y3': y3,
            'y4': y4,
            'y5': y5,
        }
        