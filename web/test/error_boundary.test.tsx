import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "../src/components/ErrorBoundary";

function Boom({ fail }: { fail: boolean }) {
  if (fail) throw new Error("kaboom");
  return <p>fine</p>;
}

describe("U16 a crashing view is contained and explained, not a blank page", () => {
  it("shows the error message and lets the user try again", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    function Harness() {
      const [fail, setFail] = useState(true);
      return (
        <>
          <button type="button" onClick={() => setFail(false)}>
            fix
          </button>
          <ErrorBoundary label="Results">
            <Boom fail={fail} />
          </ErrorBoundary>
        </>
      );
    }
    const user = userEvent.setup();
    render(<Harness />);
    const alert = screen.getByTestId("view-error");
    expect(alert).toHaveTextContent("The Results view crashed.");
    expect(alert).toHaveTextContent("kaboom");
    expect(alert).toHaveTextContent("restart the server");
    await user.click(screen.getByRole("button", { name: "fix" }));
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(screen.getByText("fine")).toBeInTheDocument();
    spy.mockRestore();
  });
});
