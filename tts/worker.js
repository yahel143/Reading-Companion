// Runs the voice model off the main thread so the page never freezes while speech is generated.
const ORT_BASE = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.18.0/';
importScripts(ORT_BASE + 'ort.min.js');
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.numThreads = 1;   // multi-threading needs server headers GitHub Pages can't send

let session = null;

onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === 'init') {
      session = await ort.InferenceSession.create(e.data.model, { executionProviders: ['wasm'] });
      postMessage({ id, ok: true });
    } else if (type === 'run') {
      const { ids, scales } = e.data;
      const out = await session.run({
        input: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
        input_lengths: new ort.Tensor('int64', BigInt64Array.from([BigInt(ids.length)]), [1]),
        scales: new ort.Tensor('float32', Float32Array.from(scales), [3]),
      });
      const pcm = out.output.data;
      postMessage({ id, ok: true, pcm }, [pcm.buffer]);
    }
  } catch (err) {
    postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
