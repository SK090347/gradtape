import { MLP, sgdStep } from './nn.js';
import { mse, Value } from './value.js';

/**
 * Lab notebook style demo:
 * fit a tiny MLP on XOR, then a 1-d regression sanity check.
 */

const xor: [number[], number][] = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0],
];

function trainXor(epochs = 800, lr = 0.15): MLP {
  // fix seed-ish by just running; randomness is ok for a demo
  const net = new MLP(2, [6, 1]);

  for (let ep = 0; ep < epochs; ep++) {
    let loss = new Value(0);
    for (const [x, y] of xor) {
      const pred = net.forward(x)[0]!;
      loss = loss.add(mse(pred, y));
    }
    loss = loss.mul(1 / xor.length);
    loss.backward();
    sgdStep(net.params(), lr);

    if (ep % 200 === 0 || ep === epochs - 1) {
      console.log(`epoch ${ep}  loss=${loss.data.toFixed(5)}`);
    }
  }
  return net;
}

function showXor(net: MLP): void {
  console.log('\nXOR preds (want ~0,1,1,0):');
  for (const [x, y] of xor) {
    const p = net.forward(x)[0]!.data;
    console.log(`  ${x} -> ${p.toFixed(3)}  (target ${y})`);
  }
}

function regressionToy(): void {
  // fit y ≈ 2x + 1 with one linear-ish neuron path (tanh bottleneck, still learns slope-ish)
  console.log('\n--- synthetic y=2x+1 (quick) ---');
  const net = new MLP(1, [8, 1]);
  const xs = [-1, -0.5, 0, 0.5, 1];
  for (let ep = 0; ep < 400; ep++) {
    let loss = new Value(0);
    for (const x of xs) {
      const y = 2 * x + 1;
      // scale target into (-1,1) so tanh head can reach it
      const target = Math.tanh(y / 3);
      loss = loss.add(mse(net.forward([x])[0]!, target));
    }
    loss = loss.mul(1 / xs.length);
    loss.backward();
    sgdStep(net.params(), 0.2);
    if (ep % 100 === 0) console.log(`epoch ${ep}  loss=${loss.data.toFixed(5)}`);
  }
}

console.log('gradtape demo — reverse-mode tape + tiny MLP\n');
const net = trainXor();
showXor(net);
regressionToy();
console.log('\ndone.');
