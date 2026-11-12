import * as Tooltip from "@radix-ui/react-tooltip";
import { useQuery } from "@tanstack/react-query";
import {
  type InputHTMLAttributes,
  type KeyboardEvent,
  type ReactNode,
  type Ref,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { api } from "../api/client";
import type { Glossary } from "../api/types";
import { useTruth } from "../state/truth";
import { useUI } from "../state/ui";
import { fmt, parseNumber } from "../state/worldOps";

// ----------------------------------------------------------------------------- tooltip

export function Tip({
  content,
  children,
  side = "top",
}: {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
}) {
  // Radix closes a tooltip when any ancestor of its trigger scrolls. Focusing a trigger inside a
  // scrollable panel scrolls it into view, which would close the tooltip at once; so while the trigger
  // has keyboard focus we keep it open. Escape still dismisses it (WCAG 1.4.13).
  const [open, setOpen] = useState(false);
  const focused = useRef(false);
  const escaped = useRef(false);
  return (
    <Tooltip.Root
      open={open}
      onOpenChange={(o) => {
        if (!o && focused.current && !escaped.current) return;
        escaped.current = false;
        setOpen(o);
      }}
    >
      <Tooltip.Trigger
        asChild
        onFocus={() => {
          focused.current = true;
        }}
        onBlur={() => {
          focused.current = false;
        }}
      >
        {children}
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          className="tip"
          side={side}
          sideOffset={5}
          collisionPadding={8}
          onEscapeKeyDown={() => {
            escaped.current = true;
          }}
        >
          {content}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

// ----------------------------------------------------------------------------- glossary terms (UX5)

export function useGlossary(): Glossary | undefined {
  return useQuery({ queryKey: ["glossary"], queryFn: api.glossary, staleTime: Number.POSITIVE_INFINITY })
    .data;
}

export function glossaryText(g: Glossary | undefined, term: string): string {
  const e = g?.[term];
  if (!e) return "";
  return e.amendment ? `${e.definition} ${e.amendment}` : e.definition;
}

/** A word with its glossary definition as tooltip. `term` must be a key of /api/glossary. */
export function Term({
  term,
  children,
  className = "",
}: {
  term: string;
  children?: ReactNode;
  className?: string;
}) {
  const g = useGlossary();
  const text = glossaryText(g, term);
  return (
    <Tip content={text || term}>
      {/* biome-ignore lint/a11y/noNoninteractiveTabindex: focusable so the definition is reachable by keyboard */}
      <span tabIndex={0} className={`term ${className}`} data-glossary={term}>
        {children ?? term}
      </span>
    </Tip>
  );
}

// ----------------------------------------------------------------------------- derived values

/**
 * A read-only number computed by the server (never typed). Styled differently from inputs, animates
 * when it changes, carries a ripple marker (U3) and explains itself on hover (label + source).
 */
export function DerivedValue({
  value,
  label,
  source,
  rippleKey,
  digits = 3,
  testId,
  extra,
}: {
  value: number | null | undefined;
  label: string;
  source?: string;
  rippleKey?: string;
  digits?: number;
  testId?: string;
  extra?: ReactNode;
}) {
  const { hash, stale } = useTruth();
  const ripple = useUI((s) => (rippleKey ? s.ripple.kinds[rippleKey] : undefined));
  const seq = useUI((s) => s.ripple.seq);
  const prev = useRef(value);
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    if (prev.current !== value && prev.current !== undefined) {
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 900);
      prev.current = value;
      return () => clearTimeout(t);
    }
    prev.current = value;
  }, [value]);
  const text = fmt(value ?? null, digits);
  const src = source ?? `Derived from the World by the analytic truth engine${hash ? ` · spec ${hash}` : ""}`;
  return (
    <Tip
      content={
        <div>
          <div style={{ fontWeight: 600 }}>{label}</div>
          <div>{src}</div>
          {stale && !source ? <div>Stale: the World currently has issues.</div> : null}
          {extra}
        </div>
      }
    >
      <output
        key={ripple ? `r${seq}` : "v"}
        // biome-ignore lint/a11y/noNoninteractiveTabindex: focusable so label and source are reachable by keyboard
        tabIndex={0}
        className="derived"
        data-derived=""
        data-testid={testId}
        data-ripple={ripple}
        data-flash={flash ? "on" : undefined}
        data-value={value ?? ""}
        aria-label={`${label}: ${text}`}
        style={stale && !source ? { opacity: 0.55 } : undefined}
      >
        {text}
      </output>
    </Tip>
  );
}

// ----------------------------------------------------------------------------- inputs

type FieldProps = Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "ref"> & {
  label: string;
  hideLabel?: boolean;
  error?: string | null;
  inputRef?: Ref<HTMLInputElement>;
  autoSelect?: boolean;
};

/** Numeric input that commits on Enter or blur (one undo step per commit). Escape reverts. */
export function NumberField({
  value,
  onCommit,
  label,
  hideLabel,
  error,
  allowEmpty = false,
  min,
  max,
  inputRef,
  autoSelect,
  ...rest
}: FieldProps & {
  value: number | undefined | null;
  onCommit: (v: number | null) => void;
  allowEmpty?: boolean;
  min?: number;
  max?: number;
}) {
  const id = useId();
  const shown = value === undefined || value === null ? "" : String(value);
  const [draft, setDraft] = useState(shown);
  const [bad, setBad] = useState<string | null>(null);
  useEffect(() => setDraft(shown), [shown]);
  const commit = () => {
    if (draft.trim() === "") {
      if (allowEmpty) {
        setBad(null);
        if (shown !== "") onCommit(null);
      } else {
        setBad("enter a number");
      }
      return;
    }
    const v = parseNumber(draft);
    if (v === null) return setBad("not a number");
    if (min !== undefined && v < min) return setBad(`must be ≥ ${min}`);
    if (max !== undefined && v > max) return setBad(`must be ≤ ${max}`);
    setBad(null);
    if (v !== value) onCommit(v);
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    } else if (e.key === "Escape") {
      setDraft(shown);
      setBad(null);
    }
  };
  const err = bad ?? error ?? null;
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <label htmlFor={id} className={hideLabel ? "sr-only" : "label"}>
        {label}
      </label>
      <input
        id={id}
        ref={inputRef}
        className="field num-field"
        inputMode="decimal"
        autoComplete="off"
        value={draft}
        aria-invalid={err ? true : undefined}
        aria-describedby={err ? `${id}-err` : undefined}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKey}
        onBlur={commit}
        onFocus={autoSelect ? (e) => e.currentTarget.select() : undefined}
        {...rest}
      />
      {err ? (
        <div id={`${id}-err`} role="alert" className="text-[11.5px] text-bad">
          {err}
        </div>
      ) : null}
    </div>
  );
}

/** Text input that commits on Enter or blur. `validate` returns an error message or null. */
export function TextField({
  value,
  onCommit,
  label,
  hideLabel,
  validate,
  inputRef,
  autoSelect,
  error,
  ...rest
}: FieldProps & {
  value: string;
  onCommit: (v: string) => void;
  validate?: (v: string) => string | null;
}) {
  const id = useId();
  const [draft, setDraft] = useState(value);
  const [bad, setBad] = useState<string | null>(null);
  useEffect(() => setDraft(value), [value]);
  const commit = () => {
    const v = draft.trim();
    if (v === value) return setBad(null);
    const problem = validate?.(v) ?? null;
    if (problem) return setBad(problem);
    setBad(null);
    onCommit(v);
  };
  const err = bad ?? error ?? null;
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <label htmlFor={id} className={hideLabel ? "sr-only" : "label"}>
        {label}
      </label>
      <input
        id={id}
        ref={inputRef}
        className="field"
        autoComplete="off"
        spellCheck={false}
        value={draft}
        aria-invalid={err ? true : undefined}
        aria-describedby={err ? `${id}-err` : undefined}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          } else if (e.key === "Escape") {
            setDraft(value);
            setBad(null);
          }
        }}
        onBlur={commit}
        onFocus={autoSelect ? (e) => e.currentTarget.select() : undefined}
        {...rest}
      />
      {err ? (
        <div id={`${id}-err`} role="alert" className="text-[11.5px] text-bad">
          {err}
        </div>
      ) : null}
    </div>
  );
}

export function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
  hideLabel,
  testId,
  disabled,
  selectRef,
}: {
  label: string;
  value: T;
  options: (T | { value: T; label: string })[];
  onChange: (v: T) => void;
  hideLabel?: boolean;
  testId?: string;
  disabled?: boolean;
  selectRef?: Ref<HTMLSelectElement>;
}) {
  const id = useId();
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <label htmlFor={id} className={hideLabel ? "sr-only" : "label"}>
        {label}
      </label>
      <select
        id={id}
        ref={selectRef}
        className="field"
        value={value}
        data-testid={testId}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => {
          const v = typeof o === "string" ? o : o.value;
          const l = typeof o === "string" ? o : o.label;
          return (
            <option key={v} value={v}>
              {l}
            </option>
          );
        })}
      </select>
    </div>
  );
}

export function Checkbox({
  label,
  checked,
  onChange,
  testId,
  children,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  testId?: string;
  children?: ReactNode;
}) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-1.5 text-[13px]">
      <input
        type="checkbox"
        checked={checked}
        data-testid={testId}
        onChange={(e) => onChange(e.target.checked)}
        className="h-3.5 w-3.5 accent-[var(--accent)]"
      />
      {children ?? label}
    </label>
  );
}

export function Section({
  title,
  children,
  actions,
  testId,
}: {
  title: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
  testId?: string;
}) {
  return (
    <section className="border-line border-b px-4 py-3" data-testid={testId}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="section-title m-0">{title}</h3>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="m-0 text-[12.5px] text-muted">{children}</p>;
}
