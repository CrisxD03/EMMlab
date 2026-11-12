import { describe, expect, it } from "vitest";
import type { Preset, World } from "../src/api/types";
import { worldLayout } from "../src/features/world/layout";
import { bridge } from "./utils";

async function katsantonis(extra: Partial<World> = {}) {
  const ps = await bridge.json<Preset[]>("GET", "/api/presets");
  return { ...ps.find((p) => p.name === "katsantonis_s0")!.world, ...extra };
}

const VIEWPORTS = [
  { width: 1630, height: 1100 },
  { width: 400, height: 2400 },
  { width: 2400, height: 400 },
];

describe("U1 layout direction: forced by the World, or kept when redrawing", () => {
  it("draws left to right when the World says so, whatever the canvas shape", async () => {
    for (const viewport of VIEWPORTS) {
      const forced = await worldLayout(await katsantonis({ layout_direction: "RIGHT" }), {}, viewport);
      expect(forced.direction).toBe("RIGHT");
      const { feedback, achievement } = forced.positions;
      expect(achievement.x).toBeGreaterThan(feedback.x); // parents left of children
    }
  });

  it("keeps the current direction when the caller asks, so a display toggle does not flip it", async () => {
    const kept = await worldLayout(await katsantonis(), {}, VIEWPORTS[1], "RIGHT");
    expect(kept.direction).toBe("RIGHT");
    // the World's own setting wins over the caller's
    const own = await worldLayout(await katsantonis({ layout_direction: "DOWN" }), {}, VIEWPORTS[1], "RIGHT");
    expect(own.direction).toBe("DOWN");
  });
});
