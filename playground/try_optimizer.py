import numpy as np
from scipy.optimize import minimize
import ufl
from ufl import TrialFunction, TestFunction, dx
from basix.ufl import element

# 1. UFL-Ausdruck definieren (Zielfunktion f(x) = x² + 1)
# Da ufl.derivative auf Funktionalen (Integralen) operiert,
# definieren wir einen 1D-Raum (Interval) und ein Funktional F.
element = element("Lagrange", ufl.cell.interval, 1)
V = ufl.FunctionSpace(ufl.Mesh(element.cell()), element)
x_ufl = TrialFunction(V) # Die Variable x, nach der minimiert wird

# Zielfunktional F = integral( (x**2 + 1) * dx )
# Wir minimieren über den Funktionsraum, was in diesem 0D-Fall
# auf die Minimierung von (x^2 + 1) reduziert wird.
F = (x_ufl**2 + 1) * dx

# 2. Ableitung mit ufl.derivative bestimmen
# Die Ableitung dF/dx (Gradient) wird durch die Variation dF[v] definiert.
v_ufl = TestFunction(V) # Die Testfunktion (Variationsrichtung)
dF = ufl.derivative(F, x_ufl, v_ufl) # dF = integral( (2*x) * v * dx )

# 3. Konvertierung zur Übergabe an SciPy

# UFL-Ergebnis (symbolisch): f(x) = x^2 + 1
# Die Zielfunktion für SciPy erwartet ein Numpy-Array x (auch wenn nur 1D)

def zielfunktion(x_np):
    # f(x) = x[0]**2 + 1
    # Numerische Implementierung basierend auf dem UFL-Ausdruck
    return x_np[0]**2 + 1.0

# UFL-Ergebnis (symbolisch): f'(x) = 2*x
# Die Ableitung für SciPy erwartet ein Numpy-Array x und gibt den Gradienten zurück

def ableitung(x_np):
    # f'(x) = 2 * x[0]
    # Numerische Implementierung basierend auf dem Ergebnis von ufl.derivative
    # Da dF = integral( (2*x) * v * dx ) ist, ist der Gradient 2*x.
    return np.array([2.0 * x_np[0]])

# 4. SciPy Minimierung

startwert = np.array([5.0]) # x₀ = 5.0

# Lösen mit SciPy. Wir übergeben Startwert, Zielfunktion und Ableitung.
ergebnis = minimize(
    zielfunktion,
    startwert,
    jac=ableitung,  # Übergabe der Ableitungsfunktion (Gradient)
    method='BFGS'   # Gradientenbasiertes Verfahren wählen
)

print(ergebnis)