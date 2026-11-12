// Left rail of the World view: add buttons and a keyboard-friendly list of every World element.

import type { ReactNode } from "react";
import { Term } from "../../components/ui";
import { useDoc } from "../../state/doc";
import { useTruth } from "../../state/truth";
import { type Selection, useUI } from "../../state/ui";
import { arrowKey, fmtTyped, normKey } from "../../state/worldOps";
import { addCategorical, addContinuous, relayout } from "./actions";
import { issueTarget } from "./issues";

function same(a: Selection, b: Selection) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function Item({
  sel,
  children,
  label,
  bad,
}: {
  sel: Selection;
  children: ReactNode;
  label: string;
  bad?: boolean;
}) {
  const current = useUI((s) => s.selection);
  const select = useUI((s) => s.select);
  const active = same(current, sel);
  return (
    <li>
      <button
        type="button"
        aria-label={label}
        aria-current={active ? "true" : undefined}
        onClick={() => select(sel)}
        className={`flex w-full items-center gap-1.5 rounded-md px-2 py-1 text-left text-[12.5px] ${
          active ? "bg-accent-soft text-fg" : "hover:bg-panel-2"
        }`}
      >
        {bad ? <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full bg-bad" /> : null}
        {children}
      </button>
    </li>
  );
}

function Group({ title, children, count }: { title: ReactNode; children: ReactNode; count: number }) {
  return (
    <div className="mb-3">
      <div className="section-title mb-1 flex items-center justify-between px-2">
        <span>{title}</span>
        <span className="text-faint">{count}</span>
      </div>
      <ul className="m-0 flex list-none flex-col p-0">{children}</ul>
    </div>
  );
}

export function Outline() {
  const w = useDoc((s) => s.doc.world);
  const { issues } = useTruth();
  const badSel = issues.map((i) => JSON.stringify(issueTarget(i, w).selection));
  const isBad = (s: Selection) => badSel.includes(JSON.stringify(s));
  const latents = Object.entries(w.nodes).filter(([, n]) => (n.kind ?? "latent") === "latent");
  const contAttrs = Object.entries(w.nodes).filter(([, n]) => n.kind === "attribute");
  return (
    <nav className="scroll-y flex h-full flex-col border-line border-r bg-panel" aria-label="World outline">
      <div className="flex flex-col gap-1.5 border-line border-b p-2">
        <button type="button" className="btn justify-start" onClick={() => addContinuous("latent")}>
          + Latent
        </button>
        <button type="button" className="btn justify-start" onClick={() => addCategorical()}>
          + Categorical attribute
        </button>
        <button type="button" className="btn justify-start" onClick={() => addContinuous("attribute")}>
          + Continuous attribute
        </button>
        <button
          type="button"
          className="btn btn-ghost justify-start"
          onClick={relayout}
          title="Automatic layout (ELK)"
        >
          Tidy layout
        </button>
      </div>
      <div className="p-2">
        <Group title={<Term term="Latent">Latents</Term>} count={latents.length}>
          {latents.map(([k, n]) => (
            <Item
              key={k}
              sel={{ type: "node", id: k }}
              label={`Latent ${k}`}
              bad={isBad({ type: "node", id: k })}
            >
              <span className="truncate">{k}</span>
              {(n.level ?? "student") === "school" ? <span className="badge ml-auto">school</span> : null}
              {n.observed === false ? <span className="badge ml-auto">hidden</span> : null}
            </Item>
          ))}
        </Group>
        <Group
          title={<Term term="Attribute">Attributes</Term>}
          count={Object.keys(w.attributes).length + contAttrs.length}
        >
          {Object.entries(w.attributes).map(([k, a]) => (
            <Item
              key={k}
              sel={{ type: "attribute", id: k }}
              label={`Attribute ${k}`}
              bad={isBad({ type: "attribute", id: k })}
            >
              <span className="truncate">{k}</span>
              <span className="ml-auto flex gap-1">
                {(a.level ?? "student") === "school" ? <span className="badge">school</span> : null}
                {a.role === "decoy" ? <span className="badge badge-warn">decoy</span> : null}
              </span>
            </Item>
          ))}
          {contAttrs.map(([k, n]) => (
            <Item
              key={k}
              sel={{ type: "node", id: k }}
              label={`Continuous attribute ${k}`}
              bad={isBad({ type: "node", id: k })}
            >
              <span className="truncate">{k}</span>
              <span className="ml-auto flex gap-1">
                <span className="badge">numeric</span>
                {n.role === "decoy" ? <span className="badge badge-warn">decoy</span> : null}
              </span>
            </Item>
          ))}
        </Group>
        <Group title={<Term term="Arrow">Arrows</Term>} count={w.arrows.length}>
          {w.arrows.map((a, i) => {
            const k = arrowKey(a);
            const tw = w.twists.filter((t) => normKey(t.arrow) === k).length;
            return (
              <Item
                key={`${k}-${i}`}
                sel={{ type: "arrow", id: k }}
                label={`Arrow ${k}`}
                bad={isBad({ type: "arrow", id: k })}
              >
                <span className="truncate">
                  {a.from} → {a.to}
                </span>
                <span className="num ml-auto text-muted">{fmtTyped(a.slope ?? a.std)}</span>
                {tw ? <span className="badge badge-twist">⤳{tw}</span> : null}
              </Item>
            );
          })}
        </Group>
        <Group title={<Term term="Level Shift">Level shifts</Term>} count={w.shifts.length}>
          {w.shifts.map((s, i) => (
            <Item
              key={i}
              sel={{ type: "shift", index: i }}
              label={`Level shift ${i + 1}`}
              bad={isBad({ type: "shift", index: i })}
            >
              <span className="truncate">
                {Object.values(s.when).join("&")} → {s.target}
              </span>
              <span className="num ml-auto text-muted">{fmtTyped(s.delta)}</span>
            </Item>
          ))}
        </Group>
        <Group title={<Term term="Twist">Twists</Term>} count={w.twists.length}>
          {w.twists.map((t, i) => (
            <Item
              key={i}
              sel={{ type: "twist", index: i }}
              label={`Twist ${i + 1} on ${t.arrow}`}
              bad={isBad({ type: "twist", index: i })}
            >
              <span className="truncate">{t.label ?? t.arrow}</span>
              <span className="num ml-auto text-muted">{fmtTyped(t.delta)}</span>
            </Item>
          ))}
        </Group>
      </div>
    </nav>
  );
}
