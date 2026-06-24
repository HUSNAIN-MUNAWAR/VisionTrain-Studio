import { spawn } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const root = process.cwd();
const screenshotDir = resolve(root, "docs", "screenshots");
const browserCandidates = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
].filter(Boolean);

const browserPath = browserCandidates.find((candidate) => existsSync(candidate));
if (!browserPath) {
  throw new Error("Chrome or Edge was not found. Set CHROME_PATH to a Chromium-based browser executable.");
}

mkdirSync(screenshotDir, { recursive: true });

const port = 9333 + Math.floor(Math.random() * 500);
const userDataDir = resolve(tmpdir(), `visiontrain-readme-shots-${Date.now()}`);
const browser = spawn(
  browserPath,
  [
    "--headless=new",
    "--disable-gpu",
    "--hide-scrollbars",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    "about:blank",
  ],
  { stdio: "ignore", windowsHide: true },
);

function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

async function stopBrowser() {
  browser.kill();
  await sleep(500);
  try {
    rmSync(userDataDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 250 });
  } catch (error) {
    console.warn(`Warning: could not fully remove temp browser profile: ${error.message}`);
  }
}

async function waitForJson(url, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return response.json();
    } catch {
      // Browser is still starting.
    }
    await sleep(250);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function createCdpClient(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let nextId = 1;
  const pending = new Map();
  const events = [];

  ws.addEventListener("message", (message) => {
    const payload = JSON.parse(message.data);
    if (payload.id && pending.has(payload.id)) {
      const { resolve: resolveCall, reject } = pending.get(payload.id);
      pending.delete(payload.id);
      if (payload.error) reject(new Error(payload.error.message || JSON.stringify(payload.error)));
      else resolveCall(payload.result || {});
      return;
    }
    events.push(payload);
  });

  return {
    ready: new Promise((resolveReady, rejectReady) => {
      ws.addEventListener("open", resolveReady, { once: true });
      ws.addEventListener("error", rejectReady, { once: true });
    }),
    send(method, params = {}, sessionId = undefined) {
      const id = nextId++;
      const payload = { id, method, params };
      if (sessionId) payload.sessionId = sessionId;
      ws.send(JSON.stringify(payload));
      return new Promise((resolveCall, reject) => {
        pending.set(id, { resolve: resolveCall, reject });
      });
    },
    waitForEvent(method, sessionId = undefined, timeoutMs = 10000) {
      return new Promise((resolveEvent, reject) => {
        const deadline = Date.now() + timeoutMs;
        const timer = setInterval(() => {
          const index = events.findIndex((event) => event.method === method && (!sessionId || event.sessionId === sessionId));
          if (index >= 0) {
            const [event] = events.splice(index, 1);
            clearInterval(timer);
            resolveEvent(event.params || {});
          } else if (Date.now() > deadline) {
            clearInterval(timer);
            reject(new Error(`Timed out waiting for ${method}`));
          }
        }, 50);
      });
    },
    close() {
      ws.close();
    },
  };
}

async function waitForApp(client, sessionId, expectedTitle) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const result = await client.send(
      "Runtime.evaluate",
      {
        returnByValue: true,
        expression: `(() => ({
          ready: document.readyState,
          title: document.getElementById('pageTitle')?.textContent || '',
          api: document.getElementById('apiState')?.textContent || '',
          body: document.body?.innerText?.slice(0, 2000) || ''
        }))()`,
      },
      sessionId,
    );
    const value = result.result?.value || {};
    if (value.ready === "complete" && value.title === expectedTitle && value.api === "Connected") return value;
    await sleep(250);
  }
  throw new Error(`Timed out waiting for page ${expectedTitle}`);
}

async function main() {
  try {
    const version = await waitForJson(`http://127.0.0.1:${port}/json/version`);
    const client = createCdpClient(version.webSocketDebuggerUrl);
    await client.ready;
    const { targetId } = await client.send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await client.send("Target.attachToTarget", { targetId, flatten: true });
    await client.send("Page.enable", {}, sessionId);
    await client.send("Runtime.enable", {}, sessionId);
    await client.send(
      "Emulation.setDeviceMetricsOverride",
      { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false },
      sessionId,
    );

    const shots = [
      ["overview", "Overview", "dashboard-overview.png"],
      ["hardware", "Hardware", "hardware-cockpit.png"],
      ["models", "Models", "model-registry.png"],
      ["train", "Train", "training-console.png"],
      ["export", "Export", "export-center.png"],
    ];

    for (const [hash, title, filename] of shots) {
      const loadEvent = client.waitForEvent("Page.loadEventFired", sessionId).catch(() => undefined);
      await client.send("Page.navigate", { url: `http://127.0.0.1:5173/#${hash}` }, sessionId);
      await loadEvent;
      await waitForApp(client, sessionId, title);
      await sleep(1000);
      const screenshot = await client.send("Page.captureScreenshot", { format: "png", fromSurface: true }, sessionId);
      const output = resolve(screenshotDir, filename);
      writeFileSync(output, Buffer.from(screenshot.data, "base64"));
      console.log(`Captured ${filename}`);
    }

    client.close();
  } finally {
    await stopBrowser();
  }
}

main().catch(async (error) => {
  await stopBrowser();
  console.error(error);
  process.exit(1);
});
