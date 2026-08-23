// leCore index persistence in the browser -- IndexedDB, with localStorage only where it fits.
//
// WHY INDEXEDDB AND NOT localStorage. localStorage is synchronous, string-only, and capped around
// 5 MB in every browser. A leCore index is ~1.9 MB per million tokens PACKED, so localStorage runs
// out at roughly two million tokens and blocks the main thread on the way. IndexedDB is async,
// stores ArrayBuffers without base64 inflation, and its quota is a share of free disk. Using
// localStorage for a multi-megabyte index would be the silly choice, so it is offered ONLY as a
// fallback for small bundles and it says so out loud.
//
// WHAT IS STORED: the same `lecore-index/1` bundle holographic_indexstore.py writes. One format,
// disk and browser. The postings, tf tables and document lengths are DERIVED on load -- storing
// them too would be duplicate state that can drift from the stream it describes.
//
// SELF-VERIFYING. Every load recomputes the sha256 and REFUSES a mismatch. A half-finished
// IndexedDB transaction or a quota-exceeded write leaves a payload that parses cleanly and answers
// wrongly, which is the failure mode a checksum exists for.

const DB = "lecore", STORE = "indexes", VERSION = 1;

function open() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, VERSION);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

async function sha256Hex(str) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(str));
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, "0")).join("");
}

// The digest covers the PAYLOAD only and in a fixed field order -- byte-for-byte what
// holographic_indexstore.digest() hashes, or the two sides would disagree about a valid bundle.
export async function digest(man) {
  const parts = ["format", "bits", "ntok", "ndocs", "packed", "off", "vocab"].map(k => String(man[k]));
  return sha256Hex(parts.join(""));
}

export async function put(key, man) {
  if (!man.sha256) man.sha256 = await digest(man);
  const db = await open();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(man, key);
    // Resolve on COMPLETE, not on the request's success: a request can succeed inside a
    // transaction that later aborts on quota, and resolving early reports a save that did not happen.
    tx.oncomplete = () => res(true);
    tx.onerror = tx.onabort = () => rej(tx.error || new Error("transaction aborted (quota?)"));
  });
}

export async function get(key, { verify = true } = {}) {
  const db = await open();
  const man = await new Promise((res, rej) => {
    const tx = db.transaction(STORE, "readonly");
    const r = tx.objectStore(STORE).get(key);
    r.onsuccess = () => res(r.result || null);
    r.onerror = () => rej(r.error);
  });
  if (!man) return null;
  if (verify && man.sha256) {
    const d = await digest(man);
    if (d !== man.sha256) {
      throw new Error("stored index failed its own digest -- refusing to answer from it. " +
                      "Delete the key and re-import; a partial write parses cleanly and lies.");
    }
  }
  return man;
}

export async function del(key) {
  const db = await open();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).delete(key);
    tx.oncomplete = () => res(true);
    tx.onerror = () => rej(tx.error);
  });
}

export async function list() {
  const db = await open();
  return new Promise((res, rej) => {
    const tx = db.transaction(STORE, "readonly");
    const r = tx.objectStore(STORE).getAllKeys();
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

// localStorage fallback. Deliberately capped: past ~2 MB of JSON most browsers throw
// QuotaExceededError, and a silent partial save is worse than a refusal.
const LS_LIMIT = 2 * 1024 * 1024;

export async function putLocal(key, man) {
  if (!man.sha256) man.sha256 = await digest(man);
  const s = JSON.stringify(man);
  if (s.length > LS_LIMIT) {
    throw new Error(`index is ${(s.length / 1e6).toFixed(1)} MB; localStorage caps near ` +
                    `${(LS_LIMIT / 1e6).toFixed(0)} MB. Use put() (IndexedDB) instead.`);
  }
  localStorage.setItem("lecore:" + key, s);
  return true;
}

export async function getLocal(key, { verify = true } = {}) {
  const s = localStorage.getItem("lecore:" + key);
  if (!s) return null;
  const man = JSON.parse(s);
  if (verify && man.sha256 && (await digest(man)) !== man.sha256) {
    throw new Error("stored index failed its own digest -- refusing to answer from it");
  }
  return man;
}

// ---- deriving the working index from the stored generator ----------------------------------

export function unpack(man) {
  const bin = Uint8Array.from(atob(man.packed), c => c.charCodeAt(0));
  const bits = man.bits, count = man.ntok, out = new Uint32Array(count);
  let acc = 0, nb = 0, p = 0;
  for (let i = 0; i < count; i++) {
    while (nb < bits) { acc |= bin[p++] << nb; nb += 8; }
    out[i] = acc & ((1 << bits) - 1); acc >>>= bits; nb -= bits;
  }
  return out;
}

export function u32(b64) {
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const out = new Uint32Array(bin.length / 4);
  for (let i = 0; i < out.length; i++) {
    out[i] = bin[i * 4] | (bin[i * 4 + 1] << 8) | (bin[i * 4 + 2] << 16) | (bin[i * 4 + 3] << 24);
  }
  return out >>> 0 === 0 ? out : out;   // little-endian, matching numpy '<u4'
}

/** Derive postings + document lengths from the stored stream. Milliseconds, and it guarantees the
 *  index cannot drift from the bytes that describe it. */
export function derive(man) {
  const sym = unpack(man), off = u32(man.off), vocab = u32(man.vocab);
  const N = man.ndocs, post = new Map(), dl = new Float64Array(N);
  let total = 0;
  for (let d = 0; d < N; d++) {
    const c = new Map();
    for (let i = off[d]; i < off[d + 1]; i++) {
      const h = vocab[sym[i]];
      c.set(h, (c.get(h) || 0) + 1);
    }
    for (const [h, f] of c) {
      if (!post.has(h)) post.set(h, []);
      post.get(h).push(d, f);
    }
    dl[d] = off[d + 1] - off[d];
    total += dl[d];
  }
  return { N, post, dl, avgdl: total / N, vocab, stats: man.stats || null };
}
