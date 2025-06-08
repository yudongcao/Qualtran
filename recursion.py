#!/usr/bin/env python3
import json
from collections import defaultdict
import matplotlib.pyplot as plt

class resourceEstimate:
	def __init__(self, k=3):
		self.nodes = defaultdict(int)
		self.k = k

	def _dictKey(self, d):
		return json.dumps(d, sort_keys=True)
	
	def addNode(self, node):
		key = self._dictKey(node)
		self.nodes[key] += 1

	def _getK(self, n):
		if self.k !=2 and self.k !=3:
			if n % 3 == 0:
				return 3
			else:
				return 2
		else:
			return self.k
	
	def clear(self):
		self.nodes = defaultdict(int)

	def estimate(self, n):
		b = 0
		if n == 6 or n == 5:
			node = {'type': f'base 6 node' , 'n': str(6), 'k': 'None'}
			for _ in range(0, 3):
				b += self.estimate(3)
		elif n == 4:
			node = {'type': f'base 4 node' , 'n': str(4), 'k': 'None'}
			for _ in range(0, 3):
				b += self.estimate(2)
		elif n == 3 or n == 2:
			node = {'type': f'base {n} node' , 'n': str(n), 'k': 'None'}
			b = n * 2
		else:
			k = self._getK(n)
			node = {'type': 'phaseProduct', 'n': str(n), 'k': str(k)}
			while n % k != 0:
				n += 1
			if k == 3:
				b += self.estimate(int(n/k))
				b += self.estimate(int(n/k + 2))
				b += self.estimate(int(n/k + 3))
			elif k == 2:
				b += self.estimate(int(n/k))
				b += self.estimate(int(n/k + 1))
		node['b'] = str(b)
		self.addNode(node)
		return b
			
	def printNodes(self):
		for node, ct in sorted(self.nodes.items(), key=lambda x: x[1], reverse=True):
			print(f'node={node}, ct={ct}')


ek = resourceEstimate(0)
e2 = resourceEstimate(2)
e3 = resourceEstimate(3)

kk = []
k2 = []
k3 = []
for n in range(2, 2048):
	kk.append((ek.estimate(n), n))
	k2.append((e2.estimate(n), n))
	k3.append((e3.estimate(n), n))

# Unzip tuples
bk, nk = zip(*kk)
b2, n2 = zip(*k2)
b3, n3 = zip(*k3)

plt.plot(nk, bk)
plt.plot(n2, b2)
plt.plot(n3, b3)


# Label axes
plt.xlabel('n')
plt.ylabel('b')
plt.title('b vs. n')
plt.xticks([256, 512, 1024, 2048])

plt.xlim(left=0, right=2048)
plt.ylim(bottom=0)

# Optional: grid and show
plt.grid(True)
plt.show()
