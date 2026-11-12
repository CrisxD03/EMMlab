import { screen, waitFor, within } from "@testing-library/react";
import type userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { Preset, World } from "../src/api/types";
import { useDoc } from "../src/state/doc";
import { useUI } from "../src/state/ui";
import {
  bridge,
  inspector,
  openPreset,
  outline,
  renderApp,
  settleTruth,
  typeYaml,
  yamlDocText,
} from "./utils";

type User = ReturnType<typeof userEvent.setup>;
const world = () => useDoc.getState().doc.world;
const hash = () => screen.getByTestId("world-hash").getAttribute("data-hash");

async function fill(user: User, el: Element, text: string) {
  await user.clear(el);
  await user.type(el, `${text}{Enter}`);
}

describe("U5 canvas ⇄ YAML stay in sync; import/export round-trips to an identical spec hash", () => {
  it("canvas edits appear in the YAML (comments kept) and YAML edits appear on the canvas", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("button", { name: "YAML" }));
    await waitFor(async () => expect(await yamlDocText()).toContain("feedback -> achievement"), {
      timeout: 10000,
    });
    expect(await yamlDocText()).toContain("# World A from research/04-path-model-math.md");

    // canvas -> YAML
    await user.click(outline().getByRole("button", { name: "Arrow selfeff -> achievement" }));
    await fill(user, inspector().getByLabelText("Slope"), "0.35");
    await waitFor(
      async () => expect(await yamlDocText()).toMatch(/from: selfeff, to: achievement, slope: 0\.35/),
      {
        timeout: 10000,
      },
    );
    expect(await yamlDocText()).toContain("# World A from research/04-path-model-math.md");

    // YAML -> canvas
    const text = (await yamlDocText()).replace(/(from: feedback, to: selfeff, slope: )[0-9.]+/, "$10.55");
    expect(text).toContain("slope: 0.55");
    await typeYaml(text);
    await waitFor(() => expect(world().arrows.find((a) => a.to === "selfeff")?.slope).toBe(0.55), {
      timeout: 10000,
    });
    expect(within(screen.getByTestId("edge-feedback -> selfeff")).getByText("0.55")).toBeInTheDocument();
    expect(outline().getByRole("button", { name: "Arrow feedback -> selfeff" })).toHaveTextContent("0.55");

    // invalid YAML is not applied and is reported
    await typeYaml("name: [broken");
    expect(
      await screen.findByRole("list", { name: "YAML problems" }, { timeout: 10000 }),
    ).toBeInTheDocument();
    expect(world().arrows.find((a) => a.to === "selfeff")?.slope).toBe(0.55);
  });

  it("export then import gives the identical spec hash", async () => {
    const { user } = renderApp();
    await openPreset(user, "pisa_like");
    await waitFor(() => expect(hash()).toMatch(/^[0-9a-f]{16}$/));
    const h0 = hash()!;
    const blobs: Blob[] = [];
    const orig = URL.createObjectURL;
    URL.createObjectURL = ((b: Blob) => {
      blobs.push(b);
      return "blob:x";
    }) as typeof URL.createObjectURL;
    await user.click(screen.getByRole("button", { name: "Export YAML" }));
    await waitFor(() => expect(blobs.length).toBe(1));
    URL.createObjectURL = orig;
    const text = await blobs[0].text();
    expect(text).toContain("name: pisa_like");

    // start over, then import the exported file
    useUI.getState().set({ galleryOpen: true });
    await user.click(await screen.findByRole("button", { name: "Start blank" }));
    await waitFor(() => expect(Object.keys(world().nodes)).toHaveLength(0));
    const file = new File([text], "pisa_like.yaml", { type: "text/yaml" });
    await user.upload(screen.getByTestId("import-yaml"), file);
    await waitFor(() => expect(world().name).toBe("pisa_like"), { timeout: 10000 });
    await settleTruth();
    await waitFor(() => expect(hash()).toBe(h0), { timeout: 10000 });
  });
});

// One World per validation rule of S3; each is imported as YAML, as a user would.
async function presetWorld(name: string): Promise<World> {
  const ps = await bridge.json<Preset[]>("GET", "/api/presets");
  return structuredClone(ps.find((p) => p.name === name)!.world);
}

const RULES: {
  rule: string;
  make: (w: World) => void;
  expect: (sel: ReturnType<typeof useUI.getState>["selection"]) => void;
}[] = [
  {
    rule: "a cycle",
    make: (w) => void w.arrows.push({ from: "achievement", to: "feedback", slope: 0.1 }),
    expect: (s) => expect(s).toMatchObject({ type: "arrow" }),
  },
  {
    rule: "a root with a recipe",
    make: (w) => {
      w.nodes.extra = { recipe: { intercept: 0, noise_sd: 1 } };
    },
    expect: (s) => expect(s).toEqual({ type: "node", id: "extra" }),
  },
  {
    rule: "a child without a recipe",
    make: (w) => {
      w.nodes.selfeff = { root: { mean: 0, sd: 1 } };
    },
    expect: (s) => expect(s).toEqual({ type: "node", id: "selfeff" }),
  },
  {
    rule: "a twist on a non-existent arrow",
    make: (w) => void w.twists.push({ arrow: "selfeff -> feedback", when: { ses: "rich" }, delta: 0.2 }),
    expect: (s) => expect(s).toEqual({ type: "twist", index: 1 }),
  },
  {
    rule: "a twist conditioned on a continuous attribute",
    make: (w) => {
      w.nodes.size = { kind: "attribute", root: { mean: 0, sd: 1 } };
      w.twists.push({ arrow: "feedback -> selfeff", when: { size: "1" }, delta: 0.2 });
    },
    expect: (s) => expect(s).toEqual({ type: "twist", index: 1 }),
  },
  {
    rule: "a school-level child with a student-level parent",
    make: (w) => {
      w.nodes.climate = { level: "school", recipe: { intercept: 0, noise_sd: 1 } };
      w.arrows.push({ from: "feedback", to: "climate", slope: 0.2 });
    },
    expect: (s) => expect(s).toEqual({ type: "arrow", id: "feedback -> climate" }),
  },
  {
    rule: "a categorical depends_on cycle",
    make: (w) => {
      w.attributes.a = {
        values: { u: 0.5, v: 0.5 },
        depends_on: "b",
        table: { p: { u: 0.5, v: 0.5 }, q: { u: 0.5, v: 0.5 } },
      };
      w.attributes.b = {
        values: { p: 0.5, q: 0.5 },
        depends_on: "a",
        table: { u: { p: 0.5, q: 0.5 }, v: { p: 0.5, q: 0.5 } },
      };
    },
    expect: (s) => expect(s).toMatchObject({ type: "attribute" }),
  },
  {
    rule: "probabilities not summing to 1",
    make: (w) => {
      w.attributes.ses.values = { poor: 0.5, rich: 0.6 };
    },
    expect: (s) => expect(s).toEqual({ type: "attribute", id: "ses" }),
  },
];

describe("UX6 every validation issue names its element and links to it on the canvas", () => {
  for (const r of RULES) {
    it(`S3 rule: ${r.rule}`, async () => {
      const { user } = renderApp();
      await openPreset(user, "minimal");
      const bad = await presetWorld("minimal");
      r.make(bad);
      const { text } = await bridge.json<{ text: string }>("POST", "/api/world/yaml", {
        world: bad,
        text: "",
      });
      await user.upload(
        screen.getByTestId("import-yaml"),
        new File([text], "bad.yaml", { type: "text/yaml" }),
      );
      const panel = await screen.findByTestId("issues-panel", {}, { timeout: 15000 });
      const items = within(panel).getAllByTestId("issue");
      expect(items.length).toBeGreaterThan(0);
      for (const it of items) {
        expect(within(it).getByTestId("issue-element").textContent?.trim().length).toBeGreaterThan(0);
        expect(within(it).getByRole("button", { name: /Show on canvas|Open YAML/ })).toBeInTheDocument();
      }
      await user.click(within(items[0]).getByRole("button", { name: "Show on canvas" }));
      r.expect(useUI.getState().selection);
      expect(screen.getByTestId("inspector")).not.toHaveTextContent("World settings ←");
      // the inspector shows the same problem inline, or the element it concerns
      const fix = within(items[0]).queryByRole("button", { name: /^(Remove|Normalize|Convert|Raise|Add)/ });
      if (fix) {
        await user.click(fix);
        await settleTruth();
      }
    });
  }
});
