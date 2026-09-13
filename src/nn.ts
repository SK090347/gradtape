import { Value } from './value.js';

function randn(): number {
  // Box-Muller, fine for toy init
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class Neuron {
  w: Value[];
  b: Value;

  constructor(nin: number) {
    this.w = Array.from({ length: nin }, () => new Value(randn() * 0.5));
    this.b = new Value(0);
  }

  forward(x: Value[]): Value {
    let sum = this.b;
    for (let i = 0; i < this.w.length; i++) {
      sum = sum.add(this.w[i]!.mul(x[i]!));
    }
    return sum.tanh();
  }

  params(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  neurons: Neuron[];

  constructor(nin: number, nout: number) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin));
  }

  forward(x: Value[]): Value[] {
    return this.neurons.map((n) => n.forward(x));
  }

  params(): Value[] {
    return this.neurons.flatMap((n) => n.params());
  }
}

/** Small MLP: nin → …hiddens → nout */
export class MLP {
  layers: Layer[];

  constructor(nin: number, sizes: number[]) {
    const dims = [nin, ...sizes];
    this.layers = [];
    for (let i = 0; i < sizes.length; i++) {
      this.layers.push(new Layer(dims[i]!, dims[i + 1]!));
    }
  }

  forward(x: number[] | Value[]): Value[] {
    let cur: Value[] = x.map((v) => (v instanceof Value ? v : new Value(v)));
    for (const layer of this.layers) {
      cur = layer.forward(cur);
    }
    return cur;
  }

  params(): Value[] {
    return this.layers.flatMap((l) => l.params());
  }
}

export function sgdStep(params: Value[], lr: number): void {
  for (const p of params) {
    p.data -= lr * p.grad;
  }
}
