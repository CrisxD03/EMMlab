import { useState } from "react";
import type { Issue } from "../../api/types";
import { useDoc } from "../../state/doc";
import { issueFix, issueTarget, showIssue } from "./issues";

/** Every validation issue names its element and offers "show on canvas" plus a fix when obvious (UX6). */
export function IssuesPanel({ issues, title = "World issues" }: { issues: Issue[]; title?: string }) {
  const w = useDoc((s) => s.doc.world);
  const [open, setOpen] = useState(true);
  if (!issues.length) return null;
  return (
    <section
      className="absolute right-3 bottom-3 left-3 z-10 max-h-[40%] overflow-auto rounded-lg border border-bad bg-panel shadow-lg"
      aria-label={title}
      data-testid="issues-panel"
    >
      <div className="sticky top-0 flex items-center justify-between border-line border-b bg-panel px-3 py-1.5">
        <h2 className="m-0 font-semibold text-[13px] text-bad">
          {issues.length} {issues.length === 1 ? "issue" : "issues"}: the World cannot be generated until they
          are fixed
        </h2>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          {open ? "Hide" : "Show"}
        </button>
      </div>
      {open ? (
        <ul className="m-0 list-none p-0">
          {issues.map((i, k) => {
            const t = issueTarget(i, w);
            const fix = issueFix(i, w);
            return (
              <li
                key={k}
                className="flex flex-wrap items-baseline gap-x-2 gap-y-1 border-line border-b px-3 py-2 text-[12.5px]"
                data-testid="issue"
              >
                <span className="font-semibold" data-testid="issue-element">
                  {t.label}
                </span>
                <code className="text-[11px] text-faint">{i.element}</code>
                <span className="min-w-0 flex-1">{i.message}</span>
                <span className="flex gap-1.5">
                  <button type="button" className="btn btn-sm" onClick={() => showIssue(i)}>
                    {t.yaml ? "Open YAML" : "Show on canvas"}
                  </button>
                  {fix ? (
                    <button type="button" className="btn btn-sm btn-primary" onClick={fix.run}>
                      {fix.label}
                    </button>
                  ) : null}
                </span>
              </li>
            );
          })}
        </ul>
      ) : null}
    </section>
  );
}
