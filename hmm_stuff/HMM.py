from importlib import reload

import numpy as np
import plotly.graph_objs as go
from plotly import offline

import forward_backward as fb
import viterbi as vi

offline.init_notebook_mode()


# Likelihood Estimatation

# Transition matrix
T = np.matrix("0.7 0.3; 0.3 0.7")
# Emission matrix
E = np.matrix("0.9 0.1; 0.2 0.8")
# Initial distribution
init = np.matrix("0.5; 0.5")
back_init = np.matrix("1; 1")
emission = "11211"

reload(fb)
f = fb.forward(E, T, emission, init)
x = list(range(0, len(emission)))
y = [a.item(0) for a in f]
t1 = go.Scatter(x=x, y=y)

b = fb.backward(E, T, emission, back_init)
x = list(range(len(emission)-1, -1, -1))
y = [a.item(0) for a in b]
t2 = go.Scatter(x=x, y=y)

ff = [init]+f
bb = b+[back_init]

g = fb.combine(ff, bb)
x = list(range(0, len(ff)-1))
y = [a.item(0) for a in g[1:]]
t3 = go.Scatter(x=x, y=y)

# Data shows the is rigged die probability
data = [t1, t2, t3]
offline.iplot(data)

# Decoding
# Example from Figure 17.13

# Transition matrix
T = np.matrix("0.3 0.7 0; 0 0.9 0.1; 0 0 0.4")
# Emission matrix
E = np.matrix("0.5 0.3 0.2 0 0 0 0; 0 0 0.2 0.7 0.1 0 0; 0 0 0 0.1 0 0.5 0.4")

init = np.matrix("1 0 0")
emission = "02353651"

reload(vi)
d, back = vi.viterbi(T, E, emission, init)
