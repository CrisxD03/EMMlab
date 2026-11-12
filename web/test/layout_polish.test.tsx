import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Preset } from "../src/api/types";
import { plural } from "../src/features/gallery/Gallery";
import { arrowLabelWidth, layered, worldLayout } from "../src/features/world/layout";
import { useDoc } from "../src/state/doc";
import { useUI } from "../src/state/ui";
import { arrowKey } from "../src/state/worldOps";
import { bridge, inspector, openPreset, outline, renderApp } from "./utils";

type Box = { x: number; y: number; width: number; height: number };
const overlap = (a: Box, b: Box, pad = 0) =>
  a.x < b.x + b.width + pad &&
  b.x < a.x + a.width + pad &&
  a.y < b.y + b.height + pad &&
  b.y < a.y + a.height + pad;

async function presetWorld(name: string) {
  const ps = await bridge.json<Preset[]>("GET", "/api/presets");
  return ps.find((p) => p.name === name)!.world;
}

describe("U1 canvas layout: layered, attribute lane, labels clear of nodes and each other", () => {
  for (const name of ["minimal", "pisa_like"]) {
    for (const viewport of [undefined, { width: 900, height: 800 }]) {
      it(`${name}${viewport ? " (fit-to-viewport direction)" : ""}: no node/label overlaps`, async () => {
        const w = await presetWorld(name);
        const sizes = Object.fromEntries(
          [...Object.keys(w.nodes), ...Object.keys(w.attributes)].map((id) => [
            id,
            id in w.attributes ? { width: 180, height: 64 } : { width: 210, height: 100 },
          ]),
        );
        const { positions, routes, direction } = await worldLayout(w, sizes, viewport);
        const boxes: (Box & { id: string })[] = Object.entries(positions).map(([id, p]) => ({
          id,
          ...p,
          ...sizes[id],
        }));
        for (let i = 0; i < boxes.length; i++)
          for (let j = i + 1; j < boxes.length; j++)
            expect(overlap(boxes[i], boxes[j]), `${boxes[i].id} / ${boxes[j].id}`).toBe(false);
        // every arrow is routed and has a reserved label box that touches no node and no other label
        const labels = w.arrows.map((a) => ({ key: arrowKey(a), box: routes[arrowKey(a)]?.label }));
        for (const l of labels) {
          expect(l.box, l.key).toBeDefined();
          expect(l.box!.width).toBe(arrowLabelWidth(w, l.key));
          for (const b of boxes) expect(overlap(l.box!, b), `${l.key} label / ${b.id}`).toBe(false);
        }
        for (let i = 0; i < labels.length; i++)
          for (let j = i + 1; j < labels.length; j++)
            expect(overlap(labels[i].box!, labels[j].box!), `${labels[i].key} / ${labels[j].key}`).toBe(
              false,
            );
        // attributes sit in their own lane, outside the causal graph
        const graph = boxes.filter((b) => b.id in w.nodes && !(w.nodes[b.id].kind === "attribute"));
        for (const a of Object.keys(w.attributes)) {
          if (direction === "RIGHT")
            expect(positions[a].y + 64).toBeLessThan(Math.min(...graph.map((b) => b.y)));
          else expect(positions[a].x + 180).toBeLessThan(Math.min(...graph.map((b) => b.x)));
        }
      });
    }
  }

  it("results diagram layout: subgroup labels clear of nodes and each other", async () => {
    const m = { latents: ["feedback", "selfeff", "enjoyment", "interest", "achievement"] };
    const arrows = [
      ["feedback", "selfeff"],
      ["feedback", "enjoyment"],
      ["feedback", "interest"],
      ["selfeff", "enjoyment"],
      ["enjoyment", "interest"],
      ["selfeff", "achievement"],
      ["enjoyment", "achievement"],
      ["interest", "achievement"],
      ["feedback", "achievement"],
    ];
    const res = await layered(
      m.latents.map((id) => ({ id, width: 132, height: 40 })),
      arrows.map(([from, to]) => ({ id: `${from} -> ${to}`, from, to, label: { width: 212, height: 26 } })),
      { layerGap: 50, nodeGap: 34 },
    );
    const nodes = m.latents.map((id) => ({ ...res.positions[id], width: 132, height: 40 }));
    const labels = Object.values(res.routes).map((r) => r.label!);
    expect(labels).toHaveLength(9);
    for (const l of labels) for (const n of nodes) expect(overlap(l, n)).toBe(false);
    for (let i = 0; i < labels.length; i++)
      for (let j = i + 1; j < labels.length; j++) expect(overlap(labels[i], labels[j])).toBe(false);
  });

  it("pisa_like shows its 9 level shifts as node badges by default, with a toggle to draw them as edges", async () => {
    const { user } = renderApp();
    await openPreset(user, "pisa_like");
    expect(document.querySelectorAll(".react-flow__edge-shift")).toHaveLength(0);
    expect(screen.getAllByTestId(/^shift-badge-/)).toHaveLength(9);
    await user.click(screen.getByTestId("shift-badge-0"));
    expect(useUI.getState().selection).toEqual({ type: "shift", index: 0 });
    await user.click(screen.getByRole("button", { name: "Shifts as edges" }));
    await waitFor(() => expect(screen.getByTestId("shift-8")).toBeInTheDocument());
    expect(screen.queryAllByTestId(/^shift-badge-/)).toHaveLength(0);
  });
});

describe("Gallery polish", () => {
  it("initial focus is the dialog heading (no tooltip pops over the title)", async () => {
    renderApp();
    const g = await screen.findByTestId("gallery");
    await waitFor(() => expect(within(g).getByRole("heading", { name: "Start from a World" })).toHaveFocus());
    expect(screen.queryByRole("tooltip")).toBeNull();
  });

  it("pluralizes chip counts", async () => {
    expect(plural(1, "attribute")).toBe("1 attribute");
    expect(plural(2, "decoy")).toBe("2 decoys");
    expect(plural(0, "shift")).toBe("0 shifts");
    renderApp();
    const card = await screen.findByTestId("preset-simpson", {}, { timeout: 20000 });
    expect(within(card).getByText("1 arrow")).toBeInTheDocument();
    expect(within(card).getByText("1 attribute")).toBeInTheDocument();
  });
});

describe("U9 stale-run badge follows the server's World hash", () => {
  it("is absent right after Run and appears after an edit (and goes away when the edit is undone)", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByTestId("scorecard", {}, { timeout: 30000 });
    expect(screen.queryByTestId("stale-badge")).toBeNull();
    await user.click(screen.getByRole("tab", { name: "World" }));
    await user.click(outline().getByRole("button", { name: "Arrow feedback -> selfeff" }));
    const slope = inspector().getByLabelText("Slope");
    await user.clear(slope);
    await user.type(slope, "0.45{Enter}");
    await user.click(screen.getByRole("tab", { name: "Results" }));
    expect(await screen.findByTestId("stale-badge", {}, { timeout: 10000 })).toHaveTextContent(
      "World changed since this run",
    );
    useDoc.getState().undo();
    await waitFor(() => expect(screen.queryByTestId("stale-badge")).toBeNull(), { timeout: 10000 });
  });
});
