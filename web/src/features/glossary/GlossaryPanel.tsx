import * as Dialog from "@radix-ui/react-dialog";
import { glossaryText, Term, useGlossary } from "../../components/ui";
import { useUI } from "../../state/ui";

/** All glossary terms (from tool_vision.md via /api/glossary), each with its tooltip (UX5). */
export function GlossaryPanel() {
  const open = useUI((s) => s.glossaryOpen);
  const set = useUI((s) => s.set);
  const g = useGlossary();
  return (
    <Dialog.Root open={open} modal={false} onOpenChange={(o) => set({ glossaryOpen: o })}>
      <Dialog.Portal>
        <div className="fixed inset-0 z-40 bg-black/30" aria-hidden="true" />
        <Dialog.Content
          className="fixed top-0 right-0 z-50 h-full w-[min(460px,96vw)] overflow-auto border-line border-l bg-panel p-5 shadow-2xl"
          aria-describedby="glossary-desc"
          data-testid="glossary"
        >
          <Dialog.Title className="m-0 font-semibold text-[18px]">Glossary</Dialog.Title>
          <p id="glossary-desc" className="mt-1 text-[12.5px] text-muted">
            Hover or focus a dotted term anywhere in the app to see its definition.
          </p>
          <dl className="m-0">
            {Object.entries(g ?? {}).map(([term]) => (
              <div key={term} className="border-line border-b py-2">
                <dt className="font-semibold">
                  <Term term={term} />
                </dt>
                <dd className="m-0 mt-0.5 text-[12.5px] text-muted">{glossaryText(g, term)}</dd>
              </div>
            ))}
          </dl>
          <Dialog.Close asChild>
            <button
              type="button"
              className="btn btn-ghost absolute top-4 right-4"
              aria-label="Close glossary"
            >
              ✕
            </button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
