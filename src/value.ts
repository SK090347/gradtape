/**
 * Scalar node on a reverse-mode tape.
 * Each op records parents + a local backward closure.
 */

export class Value {
  data: number;
  grad = 0;
  parents: Value[];
  private _backward: () => void = () => {};
  /** debug tag, optional */
  op: string;

  constructor(data: number, parents: Value[] = [], op = '') {
    this.data = data;
    this.parents = parents;
    this.op = op;
  }

  static of(n: number): Value {
    return new Value(n);
  }

  add(other: Value | number): Value {
    const b = asValue(other);
    const out = new Value(this.data + b.data, [this, b], '+');
    out._backward = () => {
      this.grad += out.grad;
      b.grad += out.grad;
    };
    return out;
  }

  sub(other: Value | number): Value {
    return this.add(asValue(other).neg());
  }

  mul(other: Value | number): Value {
    const b = asValue(other);
    const out = new Value(this.data * b.data, [this, b], '*');
    out._backward = () => {
      this.grad += b.data * out.grad;
      b.grad += this.data * out.grad;
    };
    return out;
  }

  div(other: Value | number): Value {
    return this.mul(asValue(other).pow(-1));
  }

  neg(): Value {
    return this.mul(-1);
  }

  pow(exp: number): Value {
    const out = new Value(this.data ** exp, [this], `^${exp}`);
    out._backward = () => {
      this.grad += exp * this.data ** (exp - 1) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data > 0 ? this.data : 0, [this], 'relu');
    out._backward = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'tanh');
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], 'exp');
    out._backward = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  log(): Value {
    if (this.data <= 0) throw new Error(`log of non-positive: ${this.data}`);
    const out = new Value(Math.log(this.data), [this], 'log');
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  /** Zero grads in the subgraph, then reverse-accumulate from this node. */
  backward(): void {
    const order = topo(this);
    for (const v of order) v.grad = 0;
    this.grad = 1;
    for (let i = order.length - 1; i >= 0; i--) {
      order[i]!._backward();
    }
  }
}

function asValue(x: Value | number): Value {
  return x instanceof Value ? x : new Value(x);
}

/** Kahn-ish DFS topo over the parent DAG. */
function topo(root: Value): Value[] {
  const seen = new Set<Value>();
  const order: Value[] = [];

  function visit(v: Value): void {
    if (seen.has(v)) return;
    seen.add(v);
    for (const p of v.parents) visit(p);
    order.push(v);
  }

  visit(root);
  return order;
}

export function mse(pred: Value, target: number): Value {
  return pred.sub(target).pow(2);
}
