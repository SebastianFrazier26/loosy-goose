// Mirrors validate_bundle in site/build_bundle.py; tests/test_demo_bundle.py checks the field names.

export const LEVELS = ["light", "medium", "heavy"] as const;
export type Level = (typeof LEVELS)[number];

export type SegmentKind = "prose" | "code" | "tool_use" | "tool_result" | "thinking";
export type SupersedeReason = "edit_superseded" | "read_superseded" | "duplicate";

export interface Segment {
  id: number;
  turn: number;
  role: "user" | "assistant" | "system" | "tool";
  kind: SegmentKind;
  tokens: number;
  text: string;
  superseded: SupersedeReason | null;
  kept: Record<Level, boolean>;
}

export interface LevelResult {
  keep: number;
  budget: number;
  output_tokens: number;
  ratio: number | null;
  atom_recall_final: number;
}

export interface Source {
  dataset: string;
  url: string;
  license: string;
  revision: string;
  row: number;
}

export interface DemoTranscript {
  id: string;
  title: string;
  source: Source;
  selector: string;
  input_tokens: number;
  superseded_tokens: number;
  levels: Record<Level, LevelResult>;
  segments: Segment[];
}

export interface Stamp {
  loosy_goose: string;
  git_sha: string | null;
  select_model: string;
  atoms_version: number;
  supersede_version: number;
  novelty_version: number;
  levels: Record<Level, number>;
}

export interface Bundle {
  schema: 1;
  stamp: Stamp;
  transcripts: DemoTranscript[];
}

export async function loadBundle(url: string): Promise<Bundle> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const data: unknown = await res.json();
  if (
    typeof data !== "object" ||
    data === null ||
    (data as { schema?: unknown }).schema !== 1 ||
    !Array.isArray((data as { transcripts?: unknown }).transcripts)
  ) {
    throw new Error("bundle.json has an unexpected shape");
  }
  return data as Bundle;
}
