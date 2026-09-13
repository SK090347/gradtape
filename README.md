# gradtape


**Live demo:** https://sk090347.github.io/gradtape/
lab notes — reverse-mode autodiff on a scalar tape, in plain TypeScript.

each `Value` holds `data`, `grad`, and the parents that produced it. calling `backward()` walks the graph in reverse topo order and piles up gradients. ops covered so far: `+ * ^ relu tanh exp log` (and `sub/div/neg` on top).

## quick run

```bash
npm i
npm test
npm run demo
```

demo trains a tiny MLP on XOR, prints preds, then does a short synthetic regression pass. expect the XOR outputs to sit near `0, 1, 1, 0` after a few hundred steps — sometimes needs a lucky init, the test retries.

## files

- `src/value.ts` — tape node + mse
- `src/nn.ts` — Neuron / Layer / MLP + sgd step
- `src/demo.ts` — the notebook-ish script
- `tests/grad.test.ts` — finite-diff checks on the ops, XOR smoke

not a framework. just enough to feel how backprop moves numbers around.

MIT OR Apache-2.0 · Sumit Kumar Ta · Adamas University (CSE AI/ML)
