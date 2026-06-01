# DAVINCI Daemon Mode — Design

**Date:** 2026-05-31
**Status:** Design approved, pending spec review
**Branch target:** `develop` (additive; keeps the existing test suite green)

## Context

DAVINCI today runs analyses one-shot and synchronously: `davinci-monet run
config.yaml` builds a fresh `PipelineRunner`, executes the standard pipeline
(`LoadSourcesStage → PairingStage → StatisticsStage → PlottingStage →
ObsStatisticsStage → ObsPlottingStage → SaveResultsStage`), and exits. Runs take
**minutes to hours**. There is no way to react automatically when new data lands.

**Goal:** add a persistent **file-watching automation daemon**. It watches
configured directories, detects when newly-arrived data has fully landed, and
automatically launches the mapped DAVINCI analysis. It can be **entered and
controlled interactively** (a `daemon shell` REPL plus a `daemon top` dashboard)
or **run in the background** unattended.

The single biggest architectural constraint comes from the codebase being built
single-run-per-process. A long-lived process that runs many pipelines in-process
would accumulate matplotlib figure-manager state, leaked HDF5/NetCDF file handles
(the chronic `_cleanup_hdf5_state` / `_cleanup_context_datasets` hazard in
`runner.py`), per-run logging handlers, and global `rcParams`/`DEBUG` mutations —
compounding until memory or file-descriptor exhaustion. The design resolves this
by **never running a pipeline in the daemon process**: each run executes in a
fresh worker subprocess that exits afterward, so all leak-prone global state dies
with the child.

### Decisions locked during brainstorming

1. **Purpose: auto-run on triggers.** The daemon's job is automation, not a warm
   REPL, a manual job queue, or detach/reattach of hand-launched runs.
2. **Trigger: new data files appearing.** File-watching only — no schedule/cron,
   no manual/API/external triggers in this scope.
3. **Rules: declarative `watches.yaml` + live edits.** The file is the source of
   truth for declared rules; live add/remove/pause/resume are persisted in the
   daemon state store (not by rewriting hand-authored YAML).
4. **Run scope: per-rule, default whole-config.** Default `on_fire:
   whole_config` re-runs the mapped config as-is; opt into `on_fire:
   new_files_only` to inject just the newly-arrived paths into a named source's
   `files:`.
5. **Concurrency: serial FIFO queue, coalescing.** One isolated run at a time
   (`max_concurrent: 1`, configurable). Repeat triggers for an
   already-queued/running watch collapse into one pending run.
6. **Isolation: fresh subprocess per run.** The supervisor stays thin and never
   imports the scientific stack.
7. **Control surface: command shell + at-a-glance dashboard.** `daemon shell`
   (REPL) for control plus `daemon top` (live dashboard) for monitoring, both
   thin clients of the daemon over a local socket.
8. **Lifecycle: foreground primitive + start/stop wrapper.** `daemon serve` runs
   in the foreground; `daemon start`/`stop`/`status` background it with a PID +
   lock file and a graceful drain on stop.
9. **Settle detection: per-rule, quiescence default + optional sentinel.**
   Default fires after no new/modified events and stable size for the rule's
   `settle` window; a rule may instead/also wait on a sentinel/manifest marker.
10. **Notifications: desktop (macOS) + iCloud copy.** Plus always-on logging to
    job history. Desktop notification on completion/failure; on success, copy
    generated plots + a summary to the iCloud Claude folder.
11. **Watch mechanism: polling, no new dependency.** Periodic stat+glob, robust
    on NFS/Lustre/HPC scratch where inotify silently misses events. `watchdog`
    deliberately not added.
12. **Reach: local-only.** Control via a Unix-domain socket on the same host. The
    control layer is swappable to HTTP later if remote attach is ever needed.

### Non-goals (YAGNI)

Schedule/cron triggers; manual/API/external triggers; remote/HTTP control;
in-process pipeline execution; multi-user/auth; distributed/multi-node execution.
Each has a clean extension point (noted under §11) but none is built now.

## Architecture & process topology

```
                    ┌─────────────────────────────────────────────┐
   data lands  ───▶ │  SUPERVISOR (long-lived, thin, no sci stack) │
   in watched dir   │                                             │
                    │  watcher → queue → dispatcher → state/notify │
                    │     ▲                  │                     │
   control.sock ◀───┤  control server        ▼  spawn (one at a time)
        ▲           └──────────────────────┬──────────────────────┘
        │                                   │
  ┌─────┴──────┐                   ┌────────▼─────────┐  fresh interpreter,
  │ shell / top│                   │ WORKER subprocess │  dies after each run →
  │ status/stop│  thin clients     │ run_analysis(cfg) │  all matplotlib/HDF5/
  └────────────┘                   └───────────────────┘  logging leaks die with it
```

**Isolation invariant:** the supervisor process must not import matplotlib,
xarray, monet/monetio, or the pipeline. It only orchestrates: poll the
filesystem, debounce, queue, spawn workers, persist state, notify, serve the
control socket. Each analysis runs in a fresh worker subprocess that runs the
existing `run_analysis(config)` path and then exits. This keeps the daemon stable
for weeks regardless of how leaky any single run is.

## Components

Each is a small, single-purpose module (per the project's <500-line goal). New
package `davinci_monet/daemon/`:

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | Pydantic `DaemonConfig` + `WatchRule`; load/expand/merge `watches.yaml` | pydantic, existing config parser |
| `watcher.py` | Polling watcher (stat+glob every `poll_interval`); settle (quiescence) + sentinel detection → emits `TriggerEvent(watch, new_files)` | stdlib only |
| `queue.py` | Serial FIFO with **coalescing** (repeat triggers for an armed/running watch merge into one pending entry) | stdlib |
| `dispatcher.py` | Builds the job (per-job env snapshot, `on_fire` scope, `new_files_only` injection), spawns the worker, captures progress + exit status | stdlib subprocess |
| `worker.py` | Child entrypoint: sets env (incl. `HDF5_USE_FILE_LOCKING`), loads + optionally injects files into the config, calls `run_analysis`, streams progress as JSON lines, exits with success/fail code | existing pipeline |
| `control.py` | Unix-socket server; line-delimited JSON request/response + streaming variant; command handlers | stdlib socket/selectors |
| `client.py` | Thin client lib used by shell/top/status/stop | stdlib socket |
| `shell.py` | `daemon shell` REPL | rich, client |
| `dashboard.py` | `daemon top` live view (reuses existing Rich progress rendering) | rich, client |
| `state.py` | SQLite (stdlib) job history + watch runtime-status persistence | sqlite3 |
| `notify.py` | Desktop notification (macOS `osascript`/`terminal-notifier`) + iCloud copy + log | stdlib |
| `lifecycle.py` | PID + lock file, background double-fork, signal handlers, graceful drain | stdlib |
| `supervisor.py` | The long-lived event loop wiring watcher → queue → dispatcher → state/notify and the control server | the above |

CLI wiring lives in `cli/commands/daemon.py` — a `daemon` Typer sub-app
registered via `app.add_typer(daemon.app, name="daemon")` in `cli/app.py`,
mirroring the existing `get` sub-app.

## Config schema (`watches.yaml`)

```yaml
daemon:                              # daemon-level policy
  state_dir: ~/.davinci/daemon       # SQLite history, pid/lock, socket, daemon log
  poll_interval: 5s
  max_concurrent: 1                  # serial by default
  hdf5_file_locking: false           # daemon owns this; sets worker env
  max_settle_wait: 30m               # safety valve for never-settling files (optional)
  worker_timeout: null               # optional hard cap on a single run
  notifications:
    desktop: true
    icloud_copy: true
    icloud_dir: ~/Library/Mobile Documents/com~apple~CloudDocs/Claude

watches:
  cam_realtime:
    watch: ${DATA}/cam/incoming/*.nc
    run:   configs/asia-aq.yaml
    on_fire: whole_config            # default: re-run config as-is
    settle: 30s                      # quiescence window
    env: { DATA: /scratch/cam }      # optional per-rule env for the worker
  modis_stream:
    watch: ${DATA}/modis/*.hdf
    run:   configs/modis-aod.yaml
    on_fire: new_files_only          # inject only new paths
    inject_into: modis               # which source's files: to override
    sentinel: ${DATA}/modis/DELIVERED
    notify: [desktop]                # optional per-rule notification override
```

**Two env-expansion layers** (resolving the cross-contamination risk for
concurrent/queued submissions with different data roots):

1. `watches.yaml`'s own `${VAR}` (watch globs, config paths) expands at daemon
   load using the **daemon's** environment.
2. The *DAVINCI config's* `${VAR}` expands inside the **worker**, using the
   per-job environment = daemon env overlaid with the rule's `env:` block.

With `max_concurrent: 1` + subprocess isolation + per-rule `env:`, two watches
with different `${DATA}` roots never collide.

**Live edits & persistence:** `watches.yaml` is authoritative for *declared*
rules. Runtime mutations (`pause`/`resume`) and live-added rules persist in the
state store, never by rewriting hand-authored YAML. `daemon reload` re-reads the
file and reconciles (declared rules updated; live-added rules preserved unless
removed). `watch save <name>` optionally writes a live-added rule back to the
file.

## Control surface (command set)

One-shot CLI and in-shell forms share the same socket protocol:

```
daemon serve            # foreground (logs to stdout; tmux/systemd/HPC-friendly)
daemon start | stop     # background w/ PID+lock; stop = graceful drain
daemon status           # one-shot summary from anywhere
daemon reload           # re-read watches.yaml and reconcile
daemon shell            # REPL
daemon top              # live dashboard
daemon watch list | add | remove | pause | resume | trigger | save
daemon logs <job|watch> [--tail]
daemon history [--watch <name>] [--failed]
```

**Protocol:** line-delimited JSON over the Unix socket — request `{cmd, args}`,
response `{ok: bool, data | error}`. A streaming variant (server pushes framed
JSON events until the client disconnects) backs `top` and `logs --tail`.
Swappable to HTTP later without touching supervisor logic.

## Data flow (one trigger)

1. **Watcher** poll sees new matching file(s) → starts/refreshes a settle timer
   (or, for a sentinel rule, waits for the marker to appear).
2. On settle, emits `TriggerEvent(watch, new_files)` → **queue**. If that watch
   already has a pending or running job, **coalesce**: merge `new_files` into the
   pending entry rather than enqueuing a second run.
3. **Dispatcher** pops the next job (respecting `max_concurrent`), builds the
   per-job env and — for `new_files_only` — the `files:` override for
   `inject_into`, validates the resolved config, then spawns the **worker**.
4. **Worker** runs `run_analysis` → `PipelineRunner.run_from_config`, streaming
   progress JSON lines back over its pipe; writes the usual per-run Markdown log
   and analysis outputs; exits 0 on `PipelineResult.success`, non-zero otherwise.
5. Supervisor records submit/start/end/status/log-path/result-summary to
   **SQLite**, updates live job state (for `top`), and fires **notify** (desktop
   + iCloud copy of plots/summary on success; failure notification on error).

## State store (SQLite, in `state_dir/history.db`)

- **`jobs`** — `id`, `watch_name`, `config_path`, `on_fire`, `files` (JSON),
  `status` (`queued`/`running`/`completed`/`failed`/`skipped`), `submitted_at`,
  `started_at`, `ended_at`, `duration_s`, `exit_code`, `log_path`,
  `result_summary` (JSON), `error`.
- **`watch_status`** — `watch_name`, `enabled` (bool), `source` (`file`/`live`),
  `rule_json` (for live-added rules), `updated_at`.

History survives daemon restarts and backs `daemon history` / the dashboard's
RECENT panel.

## Lifecycle & signals

- `daemon serve` runs the supervisor in the foreground, logging to stdout.
- `daemon start` double-forks to background, writing `state_dir/daemon.pid` and
  acquiring `state_dir/daemon.lock`; redirects output to `state_dir/daemon.log`.
- **Already running** (lock held by a live PID) → friendly error naming the PID.
  **Stale lock** (PID dead) → auto-reclaim.
- `daemon stop` signals SIGTERM → **graceful drain**: stop accepting new
  triggers, let the in-flight worker finish (kill after `worker_timeout` if set),
  flush state, remove the socket + pid/lock. SIGINT in foreground behaves the
  same.

## Notifications

- **Always on:** outcome recorded to job history + daemon log.
- **Desktop (macOS):** completion/failure notification with status + brief
  summary, via `osascript`/`terminal-notifier`.
- **iCloud copy:** on success, copy generated plots + a run summary into
  `icloud_dir` (the CLAUDE.md convention), so results sync across devices.
- Per-rule `notify:` overrides the daemon-level default.

## Error handling

- **Worker crash / nonzero exit** → job marked FAILED with captured
  stderr/traceback, failure notification; the queue continues (one bad run never
  wedges the daemon).
- **Invalid config at fire time** → validated before spawn; FAILED with a clear
  config error; daemon stays up.
- **Never-settling file** (keeps growing) → stays "settling" (visible in `top`);
  `max_settle_wait` is the safety valve.
- **Sentinel never written** → watch stays armed; visible as such; never fires.
- **Socket/client errors** (disconnect mid-stream) → server tolerant; no daemon
  impact.
- **HDF5 locking** → daemon sets `HDF5_USE_FILE_LOCKING=FALSE` (config-controlled)
  in the worker env by default, owning the CLAUDE.md gotcha as policy rather than
  relying on the caller's shell.

## Testing strategy

Per CLAUDE.md testing rules:

- **Unit:** settle/quiescence + sentinel logic (temp dirs + injectable clock);
  coalescing queue; `watches.yaml` parse/merge/env-expansion; control-protocol
  round-trip; state-store CRUD; dispatcher job-building (env + injection) without
  a real run; notify (mocked `osascript`/copy).
- **Integration (through the real pipeline — CLAUDE.md rule #1):** start the
  supervisor against a temp `watches.yaml` pointing at a temp watched dir and a
  synthetic-data DAVINCI config; drop synthetic NetCDF files into the watched
  dir; assert settle fires, a worker runs `PipelineRunner.run_from_config`,
  outputs + stats CSV are produced, the job lands in history, and the notify hook
  is called. Uses the existing `tests/synthetic` generators.
- Per CLAUDE.md rule #2, the exact integration test-path design is presented for
  approval **before** writing tests, during implementation.

## Dependencies

**No new runtime dependencies.** `socket`, `selectors`, `subprocess`, `sqlite3`,
`signal`, `os`, `json` are stdlib; Rich/Typer/Pydantic are already present.
Polling needs nothing. `watchdog` is deliberately not added.

## Future extension points (not built now)

- **Watcher abstraction** emits `TriggerEvent`s; a schedule source or an
  API/sentinel-poke source could emit the same events later without touching the
  queue/dispatcher.
- **Control layer** is a swappable transport; an HTTP API (+ auth) could replace
  the Unix socket for remote attach (Approach C) without changing supervisor
  logic.
- **`max_concurrent` > 1** already contemplated; subprocess isolation makes
  bounded parallelism safe when a workstation/HPC node can spare the resources.
