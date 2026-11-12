import { Component, type ErrorInfo, type ReactNode } from "react";

/** Keeps one view's crash from blanking the whole app, and says what went wrong. */
export class ErrorBoundary extends Component<
  { children: ReactNode; label: string },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`emmlab: the ${this.props.label} view crashed`, error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div
        role="alert"
        className="m-4 max-w-3xl rounded-md border border-line bg-panel p-4"
        data-testid="view-error"
      >
        <div className="mb-1 font-semibold text-bad">The {this.props.label} view crashed.</div>
        <p className="mt-0 mb-2 text-[12.5px] text-muted">
          If you just updated emmlab, restart the server and reload the page (Cmd+Shift+R) so the page and the
          server run the same version. If it happens again, send this message:
        </p>
        <pre className="m-0 mb-2 overflow-x-auto whitespace-pre-wrap text-[12px]">
          {error.message}
          {error.stack ? `\n${error.stack.split("\n").slice(1, 6).join("\n")}` : ""}
        </pre>
        <button type="button" className="btn btn-sm" onClick={() => this.setState({ error: null })}>
          Try again
        </button>
      </div>
    );
  }
}
