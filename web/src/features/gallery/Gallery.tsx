import * as Dialog from "@radix-ui/react-dialog";
import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api } from "../../api/client";
import type { Preset } from "../../api/types";
import { Term } from "../../components/ui";
import { useUI } from "../../state/ui";
import { emptyWorld } from "../../state/worldOps";
import { loadWorld } from "../commands";

const ORDER = ["minimal", "pisa_like", "simpson", "level_shift_only", "variance_decoy"];

function stats(p: Preset) {
  const w = p.world;
  const lat = Object.values(w.nodes).filter((n) => (n.kind ?? "latent") === "latent").length;
  const attrs =
    Object.keys(w.attributes).length + Object.values(w.nodes).filter((n) => n.kind === "attribute").length;
  const decoys = [...Object.values(w.attributes), ...Object.values(w.nodes)].filter(
    (x) => x.role === "decoy",
  ).length;
  return { lat, attrs, decoys, arrows: w.arrows.length, shifts: w.shifts.length, twists: w.twists.length };
}

export function plural(n: number, word: string) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

export function Gallery() {
  const titleRef = useRef<HTMLHeadingElement>(null);
  const open = useUI((s) => s.galleryOpen);
  const set = useUI((s) => s.set);
  const presets = useQuery({
    queryKey: ["presets"],
    queryFn: api.presets,
    staleTime: Number.POSITIVE_INFINITY,
  });
  const [busy, setBusy] = useState<string | null>(null);
  const rank = (n: string) => (ORDER.includes(n) ? ORDER.indexOf(n) : ORDER.length);
  const sorted = [...(presets.data ?? [])].sort((a, b) => rank(a.name) - rank(b.name));
  const open_ = async (p: Preset, variant: "planted" | "null") => {
    setBusy(`${p.name}:${variant}`);
    try {
      const world = variant === "null" ? (await api.nullWorld(p.world)).world : p.world;
      await loadWorld(
        world,
        variant === "null" ? `${p.name} (null variant)` : p.name,
        variant === "null" ? "" : p.yaml,
      );
    } finally {
      setBusy(null);
    }
  };
  return (
    <Dialog.Root open={open} modal={false} onOpenChange={(o) => set({ galleryOpen: o })}>
      <Dialog.Portal>
        <div className="fixed inset-0 z-40 bg-black/35" aria-hidden="true" />
        <Dialog.Content
          className="fixed top-1/2 left-1/2 z-50 max-h-[88vh] w-[min(1040px,94vw)] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-xl border border-line bg-bg p-6 shadow-2xl"
          aria-describedby="gallery-desc"
          data-testid="gallery"
          // Focus the heading on open, so no tooltip pops up over it (the first focusable is a glossary term).
          onOpenAutoFocus={(e) => {
            e.preventDefault();
            titleRef.current?.focus();
          }}
        >
          <Dialog.Title ref={titleRef} tabIndex={-1} className="m-0 font-semibold text-[20px] outline-none">
            Start from a World
          </Dialog.Title>
          <p id="gallery-desc" className="mt-1 mb-4 max-w-3xl text-muted">
            Each preset is a <Term term="World" /> with known truth: open one, press <b>Run</b>, and compare
            what EMM finds with what was planted. A <Term term="Null World">null variant</Term> removes the
            twists, so every finding is a false positive.
          </p>
          {presets.isLoading ? <p className="text-muted">Loading presets…</p> : null}
          {presets.isError ? (
            <p className="text-bad">Could not load presets: {String(presets.error)}</p>
          ) : null}
          <ul className="m-0 grid list-none grid-cols-1 gap-3 p-0 md:grid-cols-2 lg:grid-cols-3">
            {sorted.map((p) => {
              const s = stats(p);
              return (
                <li key={p.name} className="card flex flex-col p-4" data-testid={`preset-${p.name}`}>
                  <h3 className="m-0 font-semibold text-[15px]">{p.name}</h3>
                  <p className="mt-1 mb-2 flex-1 text-[12.5px] text-muted">{p.description}</p>
                  <div className="mb-3 flex flex-wrap gap-1">
                    <span className="badge">{plural(s.lat, "latent")}</span>
                    <span className="badge">{plural(s.attrs, "attribute")}</span>
                    <span className="badge">{plural(s.arrows, "arrow")}</span>
                    {s.shifts ? <span className="badge">{plural(s.shifts, "shift")}</span> : null}
                    {s.twists ? <span className="badge badge-twist">{plural(s.twists, "twist")}</span> : null}
                    {s.decoys ? <span className="badge badge-warn">{plural(s.decoys, "decoy")}</span> : null}
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      className="btn btn-primary"
                      aria-label={`Open ${p.name}`}
                      disabled={!!busy}
                      onClick={() => open_(p, "planted")}
                    >
                      {busy === `${p.name}:planted` ? "Opening…" : `Open ${p.name}`}
                    </button>
                    {p.null_variant ? (
                      <button
                        type="button"
                        className="btn"
                        aria-label={`Open ${p.name} null variant`}
                        disabled={!!busy}
                        onClick={() => open_(p, "null")}
                      >
                        Null variant
                      </button>
                    ) : null}
                  </div>
                </li>
              );
            })}
            <li className="card flex flex-col border-dashed p-4" data-testid="preset-blank">
              <h3 className="m-0 font-semibold text-[15px]">Blank World</h3>
              <p className="mt-1 mb-3 flex-1 text-[12.5px] text-muted">
                Draw your own: latents, attributes, arrows, level shifts and twists, with derived values
                updating live.
              </p>
              <div>
                <button type="button" className="btn" onClick={() => loadWorld(emptyWorld(), "blank World")}>
                  Start blank
                </button>
              </div>
            </li>
          </ul>
          <Dialog.Close asChild>
            <button
              type="button"
              className="btn btn-ghost absolute top-4 right-4"
              aria-label="Close preset gallery"
            >
              ✕
            </button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
