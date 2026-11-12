// Spec editor split view (U5): CodeMirror 6 with YAML. Canvas edits are written into the text through
// the server's comment-preserving YAML writer; text edits are parsed by the server and applied to the
// World as one undoable edit.

import { yaml } from "@codemirror/lang-yaml";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { EditorState } from "@codemirror/state";
import { tags } from "@lezer/highlight";
import { basicSetup, EditorView } from "codemirror";
import { useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import type { Issue, World } from "../../api/types";
import { useDoc } from "../../state/doc";
import { yamlText } from "../../state/yamlText";
import { issueTarget, showIssue } from "./issues";

const highlight = HighlightStyle.define([
  { tag: tags.propertyName, color: "var(--accent)" },
  { tag: [tags.definition(tags.propertyName)], color: "var(--accent)" },
  { tag: tags.comment, color: "var(--faint)", fontStyle: "italic" },
  { tag: [tags.number, tags.bool, tags.null], color: "var(--derived-fg)" },
  { tag: [tags.string, tags.content], color: "var(--fg)" },
  { tag: [tags.punctuation, tags.separator], color: "var(--muted)" },
]);

export function YamlPanel() {
  const world = useDoc((s) => s.doc.world);
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const applyingRemote = useRef(false);
  const lastFromText = useRef<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [issues, setIssues] = useState<Issue[]>([]);
  const [status, setStatus] = useState<"synced" | "editing" | "error">("synced");

  // biome-ignore lint/correctness/useExhaustiveDependencies: mount the editor once
  useEffect(() => {
    if (!host.current) return;
    const v = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: yamlText.current,
        extensions: [
          basicSetup,
          yaml(),
          syntaxHighlighting(highlight),
          EditorView.lineWrapping,
          EditorView.contentAttributes.of({ "aria-label": "World specification (YAML)" }),
          EditorView.updateListener.of((u) => {
            if (!u.docChanged || applyingRemote.current) return;
            const text = u.state.doc.toString();
            yamlText.current = text;
            setStatus("editing");
            clearTimeout(timer.current);
            timer.current = setTimeout(() => void applyText(text), 450);
          }),
        ],
      }),
    });
    view.current = v;
    return () => {
      clearTimeout(timer.current);
      v.destroy();
      view.current = null;
    };
  }, []);

  async function applyText(text: string) {
    const r = await api.parse(text);
    if (!r.ok || !r.world) {
      setIssues(r.issues ?? []);
      setStatus("error");
      return;
    }
    setIssues([]);
    setStatus("synced");
    const cur = useDoc.getState().doc.world;
    const next = r.world as World;
    if (JSON.stringify(cur) === JSON.stringify(next)) return;
    lastFromText.current = JSON.stringify(next);
    useDoc.getState().apply("edit YAML", "yaml", (d) => {
      d.world = next;
    });
  }

  // Canvas -> text.
  useEffect(() => {
    const json = JSON.stringify(world);
    if (lastFromText.current === json) return;
    let cancelled = false;
    api
      .yaml(world, yamlText.current)
      .then((r) => {
        if (cancelled || !view.current) return;
        yamlText.current = r.text;
        const v = view.current;
        if (v.state.doc.toString() === r.text) return;
        applyingRemote.current = true;
        const head = Math.min(v.state.selection.main.head, r.text.length);
        v.dispatch({
          changes: { from: 0, to: v.state.doc.length, insert: r.text },
          selection: { anchor: head },
        });
        applyingRemote.current = false;
        setStatus("synced");
      })
      .catch(() => {
        /* World not representable (pydantic error): the issues panel explains it */
      });
    return () => {
      cancelled = true;
    };
  }, [world]);

  return (
    <section
      className="flex h-full min-h-0 flex-col border-line border-l bg-panel"
      aria-label="YAML editor"
      data-testid="yaml-panel"
    >
      <div className="flex items-center justify-between border-line border-b px-3 py-1.5">
        <h2 className="section-title m-0">World YAML</h2>
        <span
          className={`text-[11.5px] ${status === "error" ? "text-bad" : "text-faint"}`}
          data-testid="yaml-status"
        >
          {status === "synced"
            ? "in sync with canvas"
            : status === "editing"
              ? "applying…"
              : "not applied: fix the text"}
        </span>
      </div>
      <div ref={host} className="min-h-0 flex-1 overflow-auto" data-testid="yaml-editor" />
      {issues.length ? (
        <ul
          className="m-0 max-h-40 list-none overflow-auto border-line border-t p-2 text-[12px]"
          aria-label="YAML problems"
        >
          {issues.map((i, k) => (
            <li key={k} role="alert" className="mb-1 text-bad">
              <code>{i.element}</code>: {i.message}
              {issueTarget(i, useDoc.getState().doc.world).selection ? (
                <button type="button" className="link ml-2" onClick={() => showIssue(i)}>
                  show on canvas
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
