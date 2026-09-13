import { describe, expect, it } from 'vitest';
import { MLP, sgdStep } from '../src/nn.js';
import { mse, Value } from '../src/value.js';

/** Central finite difference on a scalar fn: R → R */
function finiteDiff(f: (x: number) => number, x: number, eps = 1e-5): number {
  return (f(x + eps) - f(x - eps)) / (2 * eps);
}

describe('ops + gradient check', () => {
  it('add/mul/pow match finite differences', () => {
    const a0 = 2.3;
    const b0 = -1.1;

    const a = new Value(a0);
    const b = new Value(b0);
    const y = a.mul(b).add(a.pow(2)).add(b);
    y.backward();

    const numA = finiteDiff((x) => x * b0 + x ** 2 + b0, a0);
    const numB = finiteDiff((x) => a0 * x + a0 ** 2 + x, b0);

    expect(a.grad).toBeCloseTo(numA, 4);
    expect(b.grad).toBeCloseTo(numB, 4);
  });

  it('relu / tanh / exp / log', () => {
    const x0 = 0.7;
    const x = new Value(x0);
    const y = x.relu().add(x.tanh()).add(x.exp()).add(x.log());
    y.backward();

    const num = finiteDiff((v) => {
      const relu = v > 0 ? v : 0;
      return relu + Math.tanh(v) + Math.exp(v) + Math.log(v);
    }, x0);

    expect(x.grad).toBeCloseTo(num, 4);
  });

  it('relu is flat on the negative side', () => {
    const x = new Value(-2);
    const y = x.relu();
    y.backward();
    expect(y.data).toBe(0);
    expect(x.grad).toBe(0);
  });

  it('mse pulls prediction toward target', () => {
    const p = new Value(0.2);
    const loss = mse(p, 1);
    loss.backward();
    expect(p.grad).toBeCloseTo(2 * (0.2 - 1), 8);
  });
});

describe('XOR mlp', () => {
  it('loss drops and preds get the right signs', () => {
    // deterministic-ish: retry a couple seeds via re-init if needed
    let ok = false;
    let lastLoss = Infinity;

    for (let attempt = 0; attempt < 3 && !ok; attempt++) {
      const net = new MLP(2, [8, 1]);
      const data: [number[], number][] = [
        [[0, 0], 0],
        [[0, 1], 1],
        [[1, 0], 1],
        [[1, 1], 0],
      ];

      let lossVal = 0;
      for (let ep = 0; ep < 1200; ep++) {
        let loss = new Value(0);
        for (const [x, y] of data) {
          loss = loss.add(mse(net.forward(x)[0]!, y));
        }
        loss = loss.mul(0.25);
        loss.backward();
        sgdStep(net.params(), 0.2);
        lossVal = loss.data;
      }

      lastLoss = lossVal;
      const preds = data.map(([x]) => net.forward(x)[0]!.data);
      // rough: near 0 for (0,0)/(1,1), near 1 for the others
      const good =
        preds[0]! < 0.35 &&
        preds[3]! < 0.35 &&
        preds[1]! > 0.65 &&
        preds[2]! > 0.65 &&
        lossVal < 0.15;
      ok = good;
    }

    expect(ok, `XOR failed to converge, last loss=${lastLoss}`).toBe(true);
  });
});
