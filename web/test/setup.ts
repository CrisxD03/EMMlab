import "@testing-library/jest-dom/vitest";
import { cleanup, configure } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, vi } from "vitest";
import { Bridge, makeEventSource } from "./bridge";

// ----------------------------------------------------------------------------- jsdom gaps
// React Flow's documented jsdom setup: ResizeObserver, DOMMatrixReadOnly and element sizes.

class ResizeObserverMock {
  cb: ResizeObserverCallback;
  constructor(cb: ResizeObserverCallback) {
    this.cb = cb;
  }
  observe(target: Element) {
    const el = target as HTMLElement;
    const width = el.offsetWidth ?? 200;
    const height = el.offsetHeight ?? 80;
    const box = [{ inlineSize: width, blockSize: height }];
    const entry = {
      target,
      contentRect: { x: 0, y: 0, top: 0, left: 0, width, height, right: width, bottom: height },
      borderBoxSize: box,
      contentBoxSize: box,
      devicePixelContentBoxSize: box,
    } as unknown as ResizeObserverEntry;
    this.cb([entry], this as unknown as ResizeObserver);
  }
  unobserve() {}
  disconnect() {}
}

class DOMMatrixReadOnlyMock {
  m22: number;
  constructor(transform?: string) {
    const scale = transform?.match(/scale\(([1-9.]+)\)/)?.[1];
    this.m22 = scale !== undefined ? +scale : 1;
  }
}

const g = globalThis as unknown as Record<string, unknown>;
g.ResizeObserver = ResizeObserverMock;
g.DOMMatrixReadOnly = DOMMatrixReadOnlyMock;
Object.defineProperties(HTMLElement.prototype, {
  offsetHeight: {
    configurable: true,
    get() {
      return Number.parseFloat((this as HTMLElement).style.height) || 80;
    },
  },
  offsetWidth: {
    configurable: true,
    get() {
      return Number.parseFloat((this as HTMLElement).style.width) || 200;
    },
  },
});
(SVGElement.prototype as unknown as { getBBox: () => DOMRect }).getBBox = () =>
  ({ x: 0, y: 0, width: 0, height: 0 }) as DOMRect;

// CodeMirror measures text ranges.
const rect = () =>
  ({ x: 0, y: 0, width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0, toJSON() {} }) as DOMRect;
Range.prototype.getBoundingClientRect = rect;
Range.prototype.getClientRects = () =>
  ({ length: 0, item: () => null, [Symbol.iterator]: [][Symbol.iterator] }) as unknown as DOMRectList;
if (!document.elementFromPoint) document.elementFromPoint = () => null;

window.matchMedia =
  window.matchMedia ??
  ((q: string) =>
    ({
      matches: false,
      media: q,
      onchange: null,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList);
Element.prototype.scrollIntoView = Element.prototype.scrollIntoView ?? (() => {});
window.requestAnimationFrame =
  window.requestAnimationFrame ?? ((cb: FrameRequestCallback) => setTimeout(() => cb(0), 0));
URL.createObjectURL = vi.fn(() => "blob:mock");
URL.revokeObjectURL = vi.fn();
window.open = vi.fn(() => null);
// Anchor clicks with href would trigger jsdom navigation errors; downloads are asserted via spies.
HTMLAnchorElement.prototype.click = vi.fn();

// ----------------------------------------------------------------------------- real backend

export const bridge = new Bridge();
g.__bridge = bridge;
g.fetch = bridge.fetch;
g.EventSource = makeEventSource(() => bridge.fetch);

configure({ asyncUtilTimeout: 5000 });

// Warm up: the first request waits for the Python app to import (a few seconds).
beforeAll(async () => {
  bridge.start();
  await bridge.request("GET", "/api/presets");
}, 120_000);
afterEach(() => cleanup());
afterAll(async () => {
  // stop any experiment still running in this bridge before shutting it down
  try {
    const jobs = await bridge.json<{ id: string; status: string }[]>("GET", "/api/experiments");
    for (const j of jobs)
      if (j.status === "running" || j.status === "queued")
        await bridge.request("POST", `/api/experiments/${j.id}/cancel`);
  } catch {
    /* ignore */
  }
  bridge.stop();
});
