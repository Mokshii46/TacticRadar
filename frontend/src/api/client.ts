export interface SinglePrediction {
  model_key: string;
  model_name: string;
  probabilities: {
    fork: number;
    pin: number;
    skewer: number;
    discoveredAttack: number;
  };
  thresholds: {
    fork: number;
    pin: number;
    skewer: number;
    discoveredAttack: number;
  };
  detected_motifs: string[];
  latency_ms: number;
  fen?: string;
}

export interface CompareResponse {
  fen: string;
  predictions: Record<string, SinglePrediction>;
  rule_baseline: {
    flags: Record<string, number>;
    counts: Record<string, number>;
    detected_motifs: string[];
  };
}

export interface ModelInfo {
  key: string;
  name: string;
  metrics: Record<string, any>;
  thresholds: Record<string, number>;
  training_time_sec: number;
  inference_time_ms: number;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  count: number;
}

export interface MotifItem {
  motif: string;
  side: string;
  attacker_square?: string;
  attacker_piece?: string;
  target_squares?: string[];
  pinning_square?: string;
  pinned_square?: string;
  target_square?: string;
  is_absolute?: boolean;
  skewer_square?: string;
  front_square?: string;
  back_square?: string;
  slider_square?: string;
  masking_square?: string;
  is_discovered_check?: boolean;
  description: string;
}

export interface MotifSquaresResponse {
  fen: string;
  motifs: {
    fork: MotifItem[];
    pin: MotifItem[];
    skewer: MotifItem[];
    discoveredAttack: MotifItem[];
  };
  counts: {
    fork: number;
    pin: number;
    skewer: number;
    discoveredAttack: number;
  };
  flags: {
    fork: number;
    pin: number;
    skewer: number;
    discoveredAttack: number;
  };
}

export interface FeatureAttribution {
  feature: string;
  value: number;
  importance: number;
  direction: 'positive' | 'negative';
}

export interface ExplainResponse {
  fen: string;
  model_used: string;
  top_features: FeatureAttribution[];
}

const API_BASE = '/api';

export const apiClient = {
  async predict(fen: string, model: string = 'ensemble'): Promise<SinglePrediction> {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, model }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Predict error: ${res.statusText}`);
    }
    return res.json();
  },

  async predictCompare(fen: string): Promise<CompareResponse> {
    const res = await fetch(`${API_BASE}/predict/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Compare error: ${res.statusText}`);
    }
    return res.json();
  },

  async getModels(): Promise<ModelsListResponse> {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
    return res.json();
  },

  async getMetrics(): Promise<Record<string, any>> {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
    return res.json();
  },

  async getMotifSquares(fen: string): Promise<MotifSquaresResponse> {
    const res = await fetch(`${API_BASE}/motif-squares?fen=${encodeURIComponent(fen)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to fetch motif squares: ${res.statusText}`);
    }
    return res.json();
  },

  async explain(fen: string, model: string = 'xgboost'): Promise<ExplainResponse> {
    const res = await fetch(`${API_BASE}/explain?fen=${encodeURIComponent(fen)}&model=${encodeURIComponent(model)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Failed to fetch explanation: ${res.statusText}`);
    }
    return res.json();
  },
};
