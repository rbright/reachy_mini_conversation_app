// Usage: node api_js_harness.mjs <api.js file URL> <scenario>; prints the scenario result as JSON.

const [apiUrl, scenario] = process.argv.slice(2);

// A fake clock makes retry timing exact, so slow CI runners cannot shift the counts.
const realSetImmediate = setImmediate;
const timers = new Map(); // id -> { at, fn }; insertion order breaks ties
let now = 0;
let nextTimerId = 1;
globalThis.setTimeout = (fn, ms = 0) => {
  timers.set(nextTimerId, { at: now + ms, fn });
  return nextTimerId++;
};
globalThis.clearTimeout = (id) => timers.delete(id);
const flush = () => new Promise((resolve) => realSetImmediate(resolve));

async function advance(ms) {
  const end = now + ms;
  await flush();
  for (;;) {
    let dueId = null;
    for (const [id, timer] of timers) {
      if (timer.at <= end && (dueId === null || timer.at < timers.get(dueId).at)) dueId = id;
    }
    if (dueId === null) break;
    const { at, fn } = timers.get(dueId);
    timers.delete(dueId);
    now = at;
    fn();
    await flush();
  }
  now = end;
}

let attempts = 0;
let refusing = true;
let lastSocket = null;
let warnings = 0;
console.warn = () => {
  warnings += 1;
};

globalThis.location = { protocol: "http:", host: "localhost" };
globalThis.document = { body: {}, activeElement: null };
globalThis.WebSocket = class {
  static OPEN = 1;

  constructor() {
    attempts += 1;
    lastSocket = this;
    this.readyState = 0;
    const refused = refusing;
    setTimeout(() => {
      if (refused) {
        this.drop();
        return;
      }
      this.readyState = WebSocket.OPEN;
      this.onopen?.();
      const event = { jsonrpc: "2.0", method: "conversation.activity", params: { state: "listening" } };
      this.onmessage?.({ data: JSON.stringify(event) });
    }, 0);
  }

  drop() {
    this.readyState = 3;
    this.onclose?.();
  }

  send() {}
};

const api = await import(apiUrl);
const events = [];
const listen = () => api.subscribe("conversation.activity", (params) => events.push(params));

if (scenario === "startup") {
  listen();
  await advance(0);
  refusing = false;
  await advance(1500);
} else if (scenario === "unsubscribe") {
  const unsubscribe = listen();
  await advance(0);
  unsubscribe();
  const before = attempts;
  await advance(5000);
  attempts -= before;
} else if (scenario === "single-chain") {
  listen();
  // Each call opens its own refused socket while the first retry is pending; a second chain would add retries.
  for (let i = 0; i < 3; i += 1) {
    await advance(200);
    api.getStatus().catch(() => {});
  }
  await advance(3500);
} else if (scenario === "outage-recovery") {
  listen();
  await advance(2500);
  refusing = false;
  await advance(1000);
  refusing = true;
  lastSocket.drop();
  await advance(2500);
} else {
  throw new Error(`unknown scenario: ${scenario}`);
}

console.log(JSON.stringify({ attempts, events: events.length, warnings }));
process.exit(0);
