import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Job } from "../src/api/types";
import { useUI } from "../src/state/ui";
import { actionCounter, bridge, openPreset, renderApp } from "./utils";

describe("U10 experiment launcher: presets, factor + levels, n_sim, live progress, cancel/resume, MCSE", () => {
  it("runs a small sweep with live progress, shows ± MCSE results and a sweep plot with bands, links the report", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("tab", { name: "Experiments" }));
    const l = await screen.findByTestId("launcher");
    await user.click(within(l).getByLabelText("current")); // both presets
    await user.click(within(l).getByRole("button", { name: "+ Factor (sweep)" }));
    await user.selectOptions(within(l).getByLabelText("Factor 1"), "world.schools.n_schools");
    await user.type(within(l).getByLabelText("Levels of factor 1 (comma-separated)"), "40, 80{Enter}");
    await user.clear(within(l).getByLabelText("Replications (n_sim)"));
    await user.type(within(l).getByLabelText("Replications (n_sim)"), "2{Enter}");
    await user.type(within(l).getByLabelText("Workers (blank = auto)"), "2{Enter}");
    await user.click(within(l).getByRole("button", { name: "Start experiment" }));
    await waitFor(() => expect(useUI.getState().experimentId).toBeTruthy(), { timeout: 15000 });
    const id = useUI.getState().experimentId!;
    const job = await bridge.json<Job>("GET", `/api/experiments/${id}`);
    expect(job.spec.presets.sort()).toEqual(["current", "recommended"]);
    expect(job.spec.factors).toEqual([{ path: "world.schools.n_schools", levels: [40, 80], linked: [] }]);
    expect(job.spec.n_sim).toBe(2);
    expect(job.total).toBe(2 * 2 * 2 * 2);

    // live progress through the event stream (EventSource)
    const seen = new Set<string>();
    await waitFor(
      () => {
        seen.add(screen.getByTestId("progress-text").textContent ?? "");
        expect(screen.getByTestId("job-status")).toHaveTextContent("done");
      },
      { timeout: 120000, interval: 100 },
    );
    expect(seen.size).toBeGreaterThan(1);
    expect(bridge.calls.some((c) => c.path === `/api/experiments/${id}/events`)).toBe(true);
    expect(screen.getByTestId("progress")).toHaveAttribute("aria-valuenow", "16");

    // results with ± MCSE and the sweep plot with bands
    await waitFor(
      () => {
        const table = screen.getByTestId("experiment-table");
        expect(table.querySelectorAll("tbody tr").length).toBe(4); // 2 conditions x 2 presets x 1 twist
        for (const c of within(table).getAllByTestId("hit-rate")) expect(c.textContent).toMatch(/±/);
        // recovered (hit whose twisted path deviates), also on the same runs without steps (b)+(c)
        for (const id of ["recovery-rate", "recovery-rate-no-prune"])
          for (const c of within(table).getAllByTestId(id)) expect(c.textContent).toMatch(/±/);
      },
      { timeout: 15000 },
    );
    const plot = await screen.findByTestId("sweep-plot");
    expect(plot.querySelector("svg")).not.toBeNull();
    expect(plot.querySelectorAll("path").length).toBeGreaterThan(0);

    // report (U11): link + self-contained download
    expect(screen.getByTestId("open-experiment-report")).toHaveAttribute(
      "href",
      `/api/experiments/${id}/report`,
    );
    const spy = vi.spyOn(URL, "createObjectURL");
    await user.click(screen.getByRole("button", { name: "Download report (.html)" }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    const html = await (spy.mock.calls.at(-1)![0] as Blob).text();
    expect(html.toLowerCase()).toMatch(/^<!doctype html>/);
    expect(html).toContain("<svg");
    expect(html).not.toMatch(/(src|href)=["']?https?:/i);
  });

  it("cancels and resumes a running experiment", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("tab", { name: "Experiments" }));
    const l = await screen.findByTestId("launcher");
    await user.clear(within(l).getByLabelText("Replications (n_sim)"));
    await user.type(within(l).getByLabelText("Replications (n_sim)"), "2000{Enter}");
    await user.type(within(l).getByLabelText("Workers (blank = auto)"), "1{Enter}");
    await user.click(within(l).getByRole("button", { name: "Start experiment" }));
    await waitFor(() => expect(screen.getByTestId("job-status")).toHaveTextContent("running"), {
      timeout: 20000,
    });
    await waitFor(
      () => expect(Number(screen.getByTestId("progress").getAttribute("aria-valuenow"))).toBeGreaterThan(0),
      {
        timeout: 60000,
      },
    );
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.getByTestId("job-status")).toHaveTextContent("cancelled"), {
      timeout: 60000,
    });
    const done = Number(screen.getByTestId("progress").getAttribute("aria-valuenow"));
    expect(done).toBeLessThan(4000);
    // completed replications are kept and the rates are shown with MCSE
    await waitFor(() => expect(screen.getByTestId("headline-hit-rate").textContent).toMatch(/±/), {
      timeout: 15000,
    });
    await user.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(screen.getByTestId("job-status")).toHaveTextContent("running"), {
      timeout: 20000,
    });
    await waitFor(
      () =>
        expect(Number(screen.getByTestId("progress").getAttribute("aria-valuenow"))).toBeGreaterThan(done),
      {
        timeout: 60000,
      },
    );
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.getByTestId("job-status")).toHaveTextContent("cancelled"), {
      timeout: 60000,
    });
  });
});

describe("UX2 task script B: pisa_like, run once, Repeat ×200, read hit rate with MCSE in ≤ 12 actions", () => {
  it("does it in a handful of actions", async () => {
    const { user } = renderApp();
    const a = actionCounter(user);
    const gallery = await screen.findByTestId("gallery");
    await a.click(await within(gallery).findByRole("button", { name: "Open pisa_like" }), "Open pisa_like");
    await waitFor(() => expect(screen.queryByTestId("gallery")).toBeNull(), { timeout: 15000 });
    await a.click(screen.getByRole("button", { name: "Run" }), "Run");
    await screen.findByTestId("scorecard", {}, { timeout: 60000 });
    await a.click(
      within(screen.getByTestId("results")).getByRole("button", { name: "Repeat ×200" }),
      "Repeat ×200",
    );
    await waitFor(() => expect(screen.getByTestId("job-detail")).toBeInTheDocument(), { timeout: 20000 });
    const id = useUI.getState().experimentId!;
    const job = await bridge.json<Job>("GET", `/api/experiments/${id}`);
    expect(job.spec.n_sim).toBe(200);
    expect(job.spec.presets).toEqual(["recommended"]);
    // the hit rate with its MCSE (rates update while replications finish)
    const hr = await screen.findByTestId("headline-hit-rate", {}, { timeout: 120000 });
    await waitFor(() => expect(hr.textContent).toMatch(/^\d\.\d{3} ± \d\.\d{3}$/), { timeout: 60000 });
    expect(screen.getByTestId("job-detail")).toHaveTextContent("feedback helps girls more");
    console.info(`UX2 actions: ${a.count} (${a.log.join(", ")}); hit rate ${hr.textContent}`);
    expect(a.count).toBeLessThanOrEqual(12);
    await user.click(screen.getByRole("button", { name: "Cancel" })); // cleanup, not part of the task
  });
});
