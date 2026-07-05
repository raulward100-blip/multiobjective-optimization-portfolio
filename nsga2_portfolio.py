import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

class PortfolioProblem(Problem):
    def __init__(self, mu, sigma, esg):
        self.mu = mu
        self.sigma = sigma
        self.esg = esg
        n = len(mu)
        super().__init__(n_var=n, n_obj=3, xl=0, xu=1, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        x_norm = x / np.sum(x, axis=1, keepdims=True)
        f1 = -np.dot(x_norm, self.mu)      # минимизация отриц. доходности
        f2 = np.dot(x_norm, self.sigma)    # риск (линейная аппроксимация)
        f3 = -np.dot(x_norm, self.esg)     # минимизация отриц. ESG
        out["F"] = np.column_stack([f1, f2, f3])

# Данные для 5 активов
mu    = np.array([28, 24, 21, 18,  8])
sigma = np.array([38, 34, 31, 28, 12])
esg   = np.array([ 3,  4,  5,  6, 10])

problem = PortfolioProblem(mu, sigma, esg)
algorithm = NSGA2(pop_size=200,
                  sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15),
                  mutation=PM(prob=0.1, eta=20),
                  eliminate_duplicates=True)

res = minimize(problem, algorithm,
               termination=('n_gen', 300), seed=1, verbose=True)
print(f"Найдено Парето-оптимальных решений: {len(res.F)}")

# Визуализация
plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)
plt.scatter(res.F[:,1], -res.F[:,0], c='blue', alpha=0.6)
plt.xlabel('Риск, %'); plt.ylabel('Доходность, %')
plt.title('Доходность – Риск'); plt.grid(True)

plt.subplot(1,3,2)
plt.scatter(-res.F[:,2], -res.F[:,0], c='blue', alpha=0.6)
plt.xlabel('ESG'); plt.ylabel('Доходность, %')
plt.title('Доходность – ESG'); plt.grid(True)

plt.subplot(1,3,3)
plt.scatter(-res.F[:,2], res.F[:,1], c='blue', alpha=0.6)
plt.xlabel('ESG'); plt.ylabel('Риск, %')
plt.title('Риск – ESG'); plt.grid(True)

plt.tight_layout()
plt.savefig('pareto_projections.png')
plt.show()
