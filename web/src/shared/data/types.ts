export type Cache = { id: string; bytes: number; rows?: number; features?: number; tenant?: string };

export type GpuFrame = {
  name: string;
  kind: string;
  hbm_bytes: number;
  true_capacity: number;
  capacity_estimate: number;
  caches: Cache[];
  host_caches: number;
  host_bytes: number;
  queued: number;
  busy: boolean;
};

export type Frame = { t: number; gpus: GpuFrame[] };

export type Window = {
  t: number;
  requests: number;
  in_slo: number;
  throttled: number;
  oom: number;
  fits: number;
  hits: number;
};

export type Erasure = {
  tenant: string;
  requested_s: number;
  datasets: number;
  verified: boolean;
  seconds: number | null;
};

export type FleetDay = {
  seed: number;
  step_s: number;
  frames: Frame[];
  windows: Window[];
  erasures: Erasure[];
  audits: { tenant: string; requested: number; residual_bytes: number }[];
};

export type Gate = { gate: string; value: number; pass: boolean };

export type DesignRow = {
  name: string;
  description: string;
  summary: Record<string, number | null>;
  ci: Record<string, number | null>;
};

export type Designs = { seeds: number[]; hours: number; gates: Gate[]; designs: DesignRow[] };

export type Choice = {
  kind: string;
  count: number;
  eur_per_hour: number;
  true_slo: number;
  meets_target: boolean;
  overpay_eur_per_hour: number | null;
};

export type AdvisorProfile = {
  profile: string;
  requests: number;
  draft: string;
  oracle: { kind: string; count: number; eur_per_hour: number } | null;
  choices: Record<string, Choice>;
};

export type Advisor = { target: number; profiles: AdvisorProfile[] };
