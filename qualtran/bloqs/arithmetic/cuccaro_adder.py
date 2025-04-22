from typing import Dict
from qualtran import Bloq, Signature, QUInt, Register, QBit, BloqBuilder, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CNOT,Toffoli
from qualtran.drawing import show_bloq

class MAJ(Bloq):

    def signature(self) -> Signature:
        return Signature([Register('c', QBit()),
                          Register('b', QBit()),
                          Register('a', QBit())])
    
    def build_composite_bloq(self, bb:BloqBuilder, c:Soquet, b:Soquet, a:Soquet) -> Dict[str,SoquetT]:
        a, b = bb.add(CNOT(), ctrl=a, target=b)
        a, c = bb.add(CNOT(), ctrl=a, target=c)
        a, b, c = bb.add(Toffoli(), ctrl=[b, c], target=a)
        return {"c": c, "b": b, "a": a}

class UMA(Bloq):

    def signature(self) -> Signature:
        return Signature([Register('c', QBit()),
                          Register('s', QBit()),
                          Register('a', QBit())])
    
    def build_composite_bloq(self, bb:BloqBuilder, c:Soquet, s:Soquet, a:Soquet) -> Dict[str,SoquetT]:
        c, s, a = bb.add(Toffoli(), ctrl=[c, s], target=a)
        c, a = bb.add(CNOT(), ctrl=a, target=c)
        c, s = bb.add(CNOT(), ctrl=c, target=s)
        return {"c": c, "s": s, "a": a}
    
class CuccaroADD(Bloq):
    
    bits: int
    
    def signature(self) -> Signature:
       return [Register('a', QUInt(self.bits)),Register('b', QUInt(self.bits + 1))]
       
    def build_composite_bloq(self, bb:BloqBuilder,)  -> Dict[str,SoquetT]:
        n = self.bits
        a = bb.split(a)[::-1]
        b = bb.split(b)[::-1]
        c = bb.allocate(dtype=QBit())
        
        for i in range(n):
            if i == 0:
                c, b[i], a[i] = bb.add(MAJ(), c=c, b=a[i], a=b[i])
            else:
                a[i-1], b[i], a[i] = bb.add(MAJ(), c=a[i-1], s=b[i], a=a[i])

        a[n-1], b[n] = bb.add(CNOT, ctrl=a[n-1], target=b[n])
        
        for i in range(n-1, -1, -1):
            if i == 0:
                c, b[i], a[i] = bb.add(UMA(), c=c, s=b[i], a=a[i])
            else:
                a[i-1], b[i], a[i] = bb.add(UMA(), c=a[i-1], s=b[i], a=a[i])
        
        bb.free(c)
        a = bb.join(a[::-1])
        b = bb.join(b[::-1])
        return {"a": a, "b": b}