import { LEVELS, loadBundle } from "./bundle";
import type { Bundle, DemoTranscript, Level, Segment } from "./bundle";
import { DOWNSTREAM, RATE_DISTORTION, REPO_URL } from "./results";

type Stop = "full" | Level;
const STOPS: readonly Stop[] = ["full", ...LEVELS];
const CLAMP_CHARS = 420;

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
};

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  el.append(...children);
  return el;
}

function applyThemeParam(): void {
  // The portfolio embeds this page in an iframe and can force a theme with ?theme=light|dark.
  const theme = new URLSearchParams(location.search).get("theme");
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
}

const fmt = (n: number): string => n.toLocaleString("en-US");

function segmentNode(seg: Segment): HTMLLIElement {
  const mono = seg.kind !== "prose" && seg.kind !== "thinking";
  const body = h("div", { class: mono ? "text mono" : "text" });
  const long = seg.text.length > CLAMP_CHARS;
  body.textContent = long ? `${seg.text.slice(0, CLAMP_CHARS)}…` : seg.text;
  const meta = h(
    "div",
    { class: "seg-meta" },
    h("span", { class: `kind k-${seg.kind}` }, seg.kind.replace("_", " ")),
    h("span", { class: "tok" }, `${fmt(seg.tokens)} tok`),
  );
  if (seg.superseded) {
    meta.append(h("span", { class: "badge" }, seg.superseded.replace("_", " ")));
  }
  const li = h("li", { class: "seg" }, meta, body);
  if (long) {
    const more = h("button", { class: "more", type: "button", "aria-expanded": "false" }, "more");
    more.addEventListener("click", () => {
      const open = more.getAttribute("aria-expanded") === "true";
      body.textContent = open ? `${seg.text.slice(0, CLAMP_CHARS)}…` : seg.text;
      more.setAttribute("aria-expanded", String(!open));
      more.textContent = open ? "more" : "less";
    });
    li.append(more);
  }
  return li;
}

class Demo {
  private current: DemoTranscript;
  private nodes: HTMLLIElement[] = [];
  private stop: Stop = "medium";

  constructor(private readonly bundle: Bundle) {
    const first = bundle.transcripts[0];
    if (!first) throw new Error("bundle has no transcripts");
    this.current = first;
    this.buildPicker();
    const slider = $<HTMLInputElement>("level");
    slider.addEventListener("input", () => {
      this.stop = STOPS[Number(slider.value)] ?? "medium";
      this.update();
    });
    $<HTMLInputElement>("hide-dropped").addEventListener("change", (e) => {
      $("transcript").classList.toggle("hide-dropped", (e.target as HTMLInputElement).checked);
    });
    this.select(first);
  }

  private buildPicker(): void {
    const picker = $("picker");
    for (const t of this.bundle.transcripts) {
      const btn = h("button", { type: "button", role: "tab", "data-id": t.id }, t.title);
      btn.addEventListener("click", () => this.select(t));
      picker.append(btn);
    }
  }

  private select(t: DemoTranscript): void {
    this.current = t;
    for (const btn of $("picker").querySelectorAll("button")) {
      btn.setAttribute("aria-selected", String(btn.dataset.id === t.id));
    }
    const list = $("transcript");
    list.replaceChildren();
    this.nodes = [];
    let lastTurn = -1;
    for (const seg of t.segments) {
      const node = segmentNode(seg);
      if (seg.turn !== lastTurn) {
        node.classList.add("turn-start");
        node.prepend(h("div", { class: `role r-${seg.role}` }, seg.role));
        lastTurn = seg.turn;
      }
      this.nodes.push(node);
      list.append(node);
    }
    const s = t.source;
    $("attribution").replaceChildren(
      "Transcript: ",
      h("a", { href: s.url, target: "_blank", rel: "noopener" }, s.dataset),
      `, row ${s.row} (revision ${s.revision.slice(0, 7)}), ${s.license} license. `,
      "An OpenHands agent run on a public open-source repository; code excerpts belong to that ",
      "repository's authors. Shown unmodified apart from long segments being clipped for display.",
    );
    this.update();
  }

  private update(): void {
    const t = this.current;
    const stop = this.stop;
    let kept = 0;
    let sup = 0;
    t.segments.forEach((seg, i) => {
      const node = this.nodes[i];
      if (!node) return;
      const isSup = stop !== "full" && seg.superseded !== null;
      const isKept = stop === "full" || seg.kept[stop];
      node.classList.toggle("sup", isSup);
      node.classList.toggle("drop", !isKept && !isSup);
      if (isKept) kept += seg.tokens;
      if (isSup) sup += seg.tokens;
    });
    const total = t.input_tokens;
    $("bar-kept").style.width = `${(100 * kept) / total}%`;
    $("bar-sup").style.width = `${(100 * sup) / total}%`;
    $("bar").setAttribute(
      "aria-label",
      `${fmt(kept)} of ${fmt(total)} tokens kept; ${fmt(sup)} superseded`,
    );
    $("stat-tokens").textContent = `${fmt(kept)} / ${fmt(total)}`;
    if (stop === "full") {
      $("stat-ratio").textContent = "1.00x";
      $("stat-recall").textContent = "1.00";
    } else {
      const lv = t.levels[stop];
      $("stat-ratio").textContent = lv.ratio === null ? "-" : `${lv.ratio.toFixed(2)}x`;
      $("stat-recall").textContent = lv.atom_recall_final.toFixed(2);
    }
  }
}

function renderResults(): void {
  const r = RATE_DISTORTION;
  const rd = h(
    "table",
    {},
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        h("th", { scope: "col" }, "rate"),
        h("th", { scope: "col" }, "atom recall"),
        h("th", { scope: "col" }, "semantic coverage"),
        h("th", { scope: "col" }, "previous default"),
      ),
    ),
    h(
      "tbody",
      {},
      ...r.rows.map((row) =>
        h(
          "tr",
          {},
          h("th", { scope: "row" }, row.rate),
          h("td", {}, row.recall.toFixed(3)),
          h("td", {}, row.coverage.toFixed(3)),
          h("td", { class: "muted" }, `${row.oldRecall.toFixed(3)} / ${row.oldCoverage.toFixed(3)}`),
        ),
      ),
    ),
  );
  const d = DOWNSTREAM;
  const qa = h(
    "table",
    {},
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        h("th", { scope: "col" }, "level"),
        h("th", { scope: "col" }, "loosy-goose"),
        h("th", { scope: "col" }, "model summary"),
      ),
    ),
    h(
      "tbody",
      {},
      ...d.rows.map((row) =>
        h(
          "tr",
          {},
          h("th", { scope: "row" }, row.level),
          h("td", {}, `${row.loosy.toFixed(2)} at ${pct(row.loosyShare)}`),
          h("td", { class: "win" }, `${row.summary.toFixed(2)} at ${pct(row.summaryShare)}`),
        ),
      ),
    ),
  );
  $("results-body").replaceChildren(
    h(
      "p",
      {},
      `Rate-distortion on ${r.transcripts} agent transcripts (public datasets plus the author's own sessions) at matched budgets. `,
      "Atom recall is the share of identifiers, paths and numbers from the final state that survive; ",
      "semantic coverage is embedding similarity to the original. ",
      `Supersession alone removes ${pct(r.supersededShare)} of tokens losslessly; selection costs `,
      `about ${r.secondsPer100k} s per 100k tokens on CPU.`,
    ),
    h("div", { class: "scroll" }, rd),
    h("p", { class: "src" }, "Source: ", h("a", { href: r.source }, r.sourceLabel)),
    h(
      "div",
      { class: "caveat" },
      h("h3", {}, "The honest caveat"),
      h(
        "p",
        {},
        `In a downstream test (Experiment I, ${d.transcripts} public transcripts) a model-written `,
        "summary beat loosy-goose on probe question answering at every level, for all four ",
        "answering models, while using fewer tokens. Accuracy below is ",
        `${d.model}'s, on probes it answers correctly from the full context; the share is the `,
        "compressed stretch's tokens against the original (a verbatim recent tail is added to every ",
        "condition). Next-step prediction was roughly level, within wide intervals. Extraction stays ",
        "deterministic, auditable and cheap, but it is not the better compressor for recall of facts.",
      ),
      h("div", { class: "scroll" }, qa),
      h("p", { class: "src" }, "Source: ", h("a", { href: d.source }, d.sourceLabel)),
    ),
  );
}

function pct(x: number): string {
  return `${(100 * x).toFixed(1)}%`;
}

function renderFooter(bundle: Bundle | null): void {
  const parts: (Node | string)[] = [
    h("a", { href: REPO_URL, target: "_blank", rel: "noopener" }, "Source on GitHub"),
    " · MIT",
  ];
  if (bundle) {
    const st = bundle.stamp;
    const sha = st.git_sha ? ` @ ${st.git_sha.slice(0, 7)}` : "";
    parts.push(
      ` · built with loosy-goose ${st.loosy_goose}${sha}, selector `,
      h("code", {}, bundle.transcripts[0]?.selector ?? ""),
      `, embeddings ${st.select_model}`,
    );
  }
  $("footer").replaceChildren(...parts);
}

async function main(): Promise<void> {
  applyThemeParam();
  renderResults();
  try {
    const bundle = await loadBundle(`${import.meta.env.BASE_URL}bundle.json`);
    new Demo(bundle);
    renderFooter(bundle);
  } catch (err) {
    $("transcript").replaceChildren(
      h("li", { class: "error" }, `Could not load the demo data: ${String(err)}`),
    );
    renderFooter(null);
  }
}

void main();
