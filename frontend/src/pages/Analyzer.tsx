import React, { useState, useEffect, useMemo } from 'react';
import { Chess, type Square } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import {
  RotateCcw,
  Shuffle,
  ArrowUpDown,
  Zap,
  Sliders,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Layers,
  Sparkles,
  AlertCircle
} from 'lucide-react';
import {
  apiClient,
  type SinglePrediction,
  type CompareResponse,
  type MotifSquaresResponse,
  type ExplainResponse
} from '../api/client';
import { SAMPLE_PUZZLES } from '../data/samplePuzzles';

const MOTIF_METADATA: Record<string, { label: string; color: string; bg: string; border: string }> = {
  fork: {
    label: 'Fork',
    color: '#f59e0b',
    bg: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    border: 'border-amber-500/40',
  },
  pin: {
    label: 'Pin',
    color: '#10b981',
    bg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    border: 'border-emerald-500/40',
  },
  skewer: {
    label: 'Skewer',
    color: '#06b6d4',
    bg: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
    border: 'border-cyan-500/40',
  },
  discoveredAttack: {
    label: 'Discovered Attack',
    color: '#8b5cf6',
    bg: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    border: 'border-purple-500/40',
  },
};

export const Analyzer: React.FC = () => {
  const [fen, setFen] = useState<string>(SAMPLE_PUZZLES[0].fen);
  const [fenInput, setFenInput] = useState<string>(SAMPLE_PUZZLES[0].fen);
  const [fenError, setFenError] = useState<string | null>(null);
  const [orientation, setOrientation] = useState<'white' | 'black'>('white');
  const [selectedModel, setSelectedModel] = useState<string>('ensemble');
  const [compareMode, setCompareMode] = useState<boolean>(false);
  const [availableModels, setAvailableModels] = useState<Array<{ key: string; name: string }>>([
    { key: 'ensemble', name: 'Soft Voting Ensemble' },
    { key: 'xgboost', name: 'XGBoost' },
    { key: 'lightgbm', name: 'LightGBM' },
    { key: 'random_forest', name: 'Random Forest' },
    { key: 'classifier_chains', name: 'Classifier Chains (RF)' },
    { key: 'mlp', name: 'PyTorch MLP (Tabular)' },
    { key: 'cnn', name: 'PyTorch ResNet CNN (8x8x12)' },
    { key: 'svm', name: 'Linear SVM (Calibrated)' },
    { key: 'logistic_regression', name: 'Logistic Regression' },
    { key: 'decision_tree', name: 'Decision Tree' },
    { key: 'knn', name: 'k-Nearest Neighbors' },
  ]);

  // Prediction states
  const [prediction, setPrediction] = useState<SinglePrediction | null>(null);
  const [compareData, setCompareData] = useState<CompareResponse | null>(null);
  const [motifGeometry, setMotifGeometry] = useState<MotifSquaresResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [showHighlights, setShowHighlights] = useState<boolean>(true);

  // Initialize chess game from FEN
  const game = useMemo(() => {
    try {
      return new Chess(fen);
    } catch {
      return new Chess();
    }
  }, [fen]);

  // Load models on mount
  useEffect(() => {
    apiClient.getModels().then(res => {
      if (res.models && res.models.length > 0) {
        setAvailableModels(res.models.map(m => ({ key: m.key, name: m.name })));
      }
    }).catch(err => console.warn('Could not load models list:', err));
  }, []);

  // Run predictions whenever FEN or model or compareMode changes
  useEffect(() => {
    let isCancelled = false;
    setLoading(true);
    setFenError(null);

    const runInference = async () => {
      try {
        new Chess(fen);
        
        if (compareMode) {
          const cData = await apiClient.predictCompare(fen);
          if (!isCancelled) setCompareData(cData);
        } else {
          const pData = await apiClient.predict(fen, selectedModel);
          if (!isCancelled) setPrediction(pData);
        }

        const geom = await apiClient.getMotifSquares(fen);
        if (!isCancelled) setMotifGeometry(geom);

        const exp = await apiClient.explain(fen, selectedModel);
        if (!isCancelled) setExplanation(exp);

      } catch (err: any) {
        if (!isCancelled) {
          setFenError(err.message || 'Invalid chess position');
        }
      } finally {
        if (!isCancelled) setLoading(false);
      }
    };

    runInference();

    return () => {
      isCancelled = true;
    };
  }, [fen, selectedModel, compareMode]);

  // Handle piece drop on board
  const onPieceDrop = (sourceSquare: Square, targetSquare: Square): boolean => {
    try {
      const gameCopy = new Chess(fen);
      const move = gameCopy.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });

      if (move === null) return false;

      const newFen = gameCopy.fen();
      setFen(newFen);
      setFenInput(newFen);
      return true;
    } catch {
      return false;
    }
  };

  // Flip board
  const handleFlip = () => {
    setOrientation(prev => (prev === 'white' ? 'black' : 'white'));
  };

  // Toggle side to move
  const handleToggleSide = () => {
    try {
      const parts = fen.split(' ');
      if (parts.length >= 2) {
        parts[1] = parts[1] === 'w' ? 'b' : 'w';
        const newFen = parts.join(' ');
        setFen(newFen);
        setFenInput(newFen);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Apply FEN Input
  const handleApplyFen = (e: React.FormEvent) => {
    e.preventDefault();
    try {
      new Chess(fenInput.trim());
      setFen(fenInput.trim());
      setFenError(null);
    } catch (err: any) {
      setFenError('Invalid FEN: ' + err.message);
    }
  };

  // Random puzzle
  const handleRandomPuzzle = () => {
    const randomIndex = Math.floor(Math.random() * SAMPLE_PUZZLES.length);
    const puz = SAMPLE_PUZZLES[randomIndex];
    setFen(puz.fen);
    setFenInput(puz.fen);
    setOrientation(puz.turn === 'w' ? 'white' : 'black');
  };

  // Reset to start
  const handleReset = () => {
    const start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    setFen(start);
    setFenInput(start);
    setOrientation('white');
  };

  // Compute custom square highlights and arrows for detected motifs
  const { customSquareStyles, customArrows } = useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};
    const arrows: Array<{ startSquare: string; endSquare: string; color: string }> = [];

    if (!showHighlights || !motifGeometry) return { customSquareStyles: styles, customArrows: arrows };

    const motifs = motifGeometry.motifs;

    // 1. Forks
    if (motifs.fork && motifs.fork.length > 0) {
      motifs.fork.forEach(f => {
        if (f.attacker_square) {
          styles[f.attacker_square] = {
            backgroundColor: 'rgba(245, 158, 11, 0.45)',
            boxShadow: 'inset 0 0 8px #f59e0b',
          };
          if (f.target_squares) {
            f.target_squares.forEach(t => {
              styles[t] = {
                backgroundColor: 'rgba(239, 68, 68, 0.35)',
                borderRadius: '6px',
              };
              arrows.push({ startSquare: f.attacker_square!, endSquare: t, color: 'rgba(245, 158, 11, 0.85)' });
            });
          }
        }
      });
    }

    // 2. Pins
    if (motifs.pin && motifs.pin.length > 0) {
      motifs.pin.forEach(p => {
        if (p.pinning_square && p.pinned_square) {
          styles[p.pinning_square] = {
            backgroundColor: 'rgba(16, 185, 129, 0.4)',
          };
          styles[p.pinned_square] = {
            backgroundColor: 'rgba(16, 185, 129, 0.45)',
            border: '2px solid #10b981',
          };
          arrows.push({ startSquare: p.pinning_square, endSquare: p.pinned_square, color: 'rgba(16, 185, 129, 0.85)' });
        }
      });
    }

    // 3. Skewers
    if (motifs.skewer && motifs.skewer.length > 0) {
      motifs.skewer.forEach(s => {
        if (s.skewer_square && s.front_square) {
          styles[s.skewer_square] = { backgroundColor: 'rgba(6, 182, 212, 0.4)' };
          styles[s.front_square] = { backgroundColor: 'rgba(6, 182, 212, 0.45)' };
          arrows.push({ startSquare: s.skewer_square, endSquare: s.front_square, color: 'rgba(6, 182, 212, 0.85)' });
        }
      });
    }

    // 4. Discovered Attacks
    if (motifs.discoveredAttack && motifs.discoveredAttack.length > 0) {
      motifs.discoveredAttack.forEach(d => {
        if (d.masking_square) {
          styles[d.masking_square] = {
            backgroundColor: 'rgba(139, 92, 246, 0.45)',
            borderRadius: '50%',
          };
        }
        if (d.slider_square && d.target_square) {
          arrows.push({ startSquare: d.slider_square, endSquare: d.target_square, color: 'rgba(139, 92, 246, 0.7)' });
        }
      });
    }

    return { customSquareStyles: styles, customArrows: arrows };
  }, [motifGeometry, showHighlights]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Top Header / Mode Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-radar-border gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold font-heading text-white tracking-tight flex items-center gap-2">
            <Zap className="w-7 h-7 text-blue-400" />
            Tactical Opportunity Analyzer
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time multi-label tactical motif detection powered by ML & deep learning.
          </p>
        </div>

        {/* Model Selector & Compare Toggle */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5">
            <Sliders className="w-4 h-4 text-slate-400" />
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              disabled={compareMode}
              className="bg-transparent text-sm text-white font-medium focus:outline-none cursor-pointer disabled:opacity-50"
            >
              {availableModels.map(m => (
                <option key={m.key} value={m.key} className="bg-slate-900 text-white">
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => setCompareMode(!compareMode)}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-xl text-sm font-medium border transition-all ${
              compareMode
                ? 'bg-purple-600/20 text-purple-300 border-purple-500/40 shadow-sm'
                : 'bg-slate-900 text-slate-300 border-slate-800 hover:border-slate-700'
            }`}
          >
            <Layers className="w-4 h-4 text-purple-400" />
            <span>Compare All Models</span>
          </button>
        </div>
      </div>

      {/* Error alert banner */}
      {fenError && (
        <div className="mt-4 p-4 rounded-xl bg-red-950/40 border border-red-500/30 flex items-center gap-3 text-red-300 text-sm">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <span>{fenError}</span>
        </div>
      )}

      {/* Main Grid: Chessboard (Left) vs Results / Explanations (Right) */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Board & Controls (lg: 6 cols) */}
        <div className="lg:col-span-6 flex flex-col items-center">
          <div className="w-full max-w-[500px] aspect-square rounded-2xl overflow-hidden shadow-2xl border border-slate-800 bg-slate-950 p-2 sm:p-3 relative">
            <Chessboard
              options={{
                position: fen,
                onPieceDrop: ({ sourceSquare, targetSquare }) => {
                  if (!targetSquare) return false;
                  return onPieceDrop(sourceSquare as Square, targetSquare as Square);
                },
                boardOrientation: orientation,
                squareStyles: customSquareStyles,
                arrows: customArrows,
                darkSquareStyle: { backgroundColor: '#2d3748' },
                lightSquareStyle: { backgroundColor: '#cbd5e1' },
                animationDurationInMs: 200,
              }}
            />

            {loading && (
              <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-[2px] flex items-center justify-center rounded-2xl z-20">
                <div className="flex items-center space-x-3 px-4 py-2 rounded-xl bg-slate-900 border border-slate-700 text-sm font-medium text-blue-400 animate-pulse">
                  <Sparkles className="w-4 h-4 animate-spin" />
                  <span>Evaluating tactics...</span>
                </div>
              </div>
            )}
          </div>

          {/* Quick Action Controls */}
          <div className="w-full max-w-[500px] mt-4 flex items-center justify-between gap-2">
            <button
              onClick={handleFlip}
              className="flex-1 py-2 px-3 rounded-lg text-xs font-semibold bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 flex items-center justify-center gap-1.5 transition-colors"
            >
              <ArrowUpDown className="w-3.5 h-3.5" />
              <span>Flip Board</span>
            </button>

            <button
              onClick={handleToggleSide}
              className="flex-1 py-2 px-3 rounded-lg text-xs font-semibold bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 flex items-center justify-center gap-1.5 transition-colors"
            >
              <div
                className={`w-2.5 h-2.5 rounded-full border ${
                  game.turn() === 'w' ? 'bg-white border-slate-300' : 'bg-black border-slate-700'
                }`}
              />
              <span>Turn: {game.turn() === 'w' ? 'White' : 'Black'}</span>
            </button>

            <button
              onClick={handleRandomPuzzle}
              className="flex-1 py-2 px-3 rounded-lg text-xs font-semibold bg-blue-600/20 border border-blue-500/30 hover:bg-blue-600/30 text-blue-300 flex items-center justify-center gap-1.5 transition-colors"
            >
              <Shuffle className="w-3.5 h-3.5" />
              <span>Random Tactic</span>
            </button>

            <button
              onClick={handleReset}
              className="p-2 rounded-lg bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
              title="Reset to Starting Position"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* FEN Input & Paste Box */}
          <form onSubmit={handleApplyFen} className="w-full max-w-[500px] mt-4">
            <div className="flex gap-2">
              <input
                type="text"
                value={fenInput}
                onChange={e => setFenInput(e.target.value)}
                placeholder="Enter position FEN..."
                className="flex-1 px-3 py-2 rounded-xl bg-slate-900/90 border border-slate-800 text-xs font-mono text-slate-200 focus:outline-none focus:border-blue-500"
              />
              <button
                type="submit"
                className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-colors"
              >
                Apply
              </button>
            </div>
          </form>

          {/* Highlight Toggle */}
          <div className="w-full max-w-[500px] mt-3 flex items-center justify-between text-xs text-slate-400 px-1">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showHighlights}
                onChange={e => setShowHighlights(e.target.checked)}
                className="rounded border-slate-700 text-blue-600 focus:ring-0"
              />
              <span>Show motif square & arrow overlays</span>
            </label>
            <span className="text-[11px] text-slate-500">Drag pieces to make moves</span>
          </div>
        </div>

        {/* Right Column: Prediction Results & Explanations (lg: 6 cols) */}
        <div className="lg:col-span-6 space-y-6">
          {/* Main Results Panel */}
          <div className="glass-panel p-6 rounded-2xl shadow-xl">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
              <h2 className="font-heading font-bold text-lg text-white flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-blue-400" />
                Detected Tactical Motifs
              </h2>
              {prediction && (
                <div className="text-xs text-slate-400 flex items-center gap-2">
                  <span className="font-mono text-emerald-400">{prediction.latency_ms} ms</span>
                  <span className="px-2 py-0.5 rounded bg-slate-800 font-medium text-slate-300">
                    {prediction.model_name}
                  </span>
                </div>
              )}
            </div>

            {/* Probability Bars per Motif */}
            <div className="mt-5 space-y-5">
              {['fork', 'pin', 'skewer', 'discoveredAttack'].map(motifKey => {
                const meta = MOTIF_METADATA[motifKey];
                const prob = prediction ? prediction.probabilities[motifKey as keyof typeof prediction.probabilities] ?? 0 : 0;
                const thresh = prediction ? prediction.thresholds[motifKey as keyof typeof prediction.thresholds] ?? 0.5 : 0.5;
                const isDetected = prob >= thresh;

                return (
                  <div key={motifKey} className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-semibold text-slate-200">{meta.label}</span>
                        {isDetected ? (
                          <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full border flex items-center gap-1 ${meta.bg}`}>
                            <CheckCircle2 className="w-3 h-3" />
                            DETECTED
                          </span>
                        ) : (
                          <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700/60 flex items-center gap-1">
                            <XCircle className="w-3 h-3" />
                            ABSENT
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-2 text-xs font-mono">
                        <span className="text-slate-400 text-[11px]">Threshold: {(thresh * 100).toFixed(0)}%</span>
                        <span className="text-white font-bold">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    {/* Progress Bar with Threshold Marker */}
                    <div className="relative w-full h-3 bg-slate-800/80 rounded-full overflow-hidden border border-slate-700/50">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.min(100, prob * 100)}%`,
                          backgroundColor: isDetected ? meta.color : '#64748b',
                        }}
                      />
                      {/* Tuned Threshold Indicator Mark */}
                      <div
                        className="absolute top-0 bottom-0 w-0.5 bg-white/70 shadow-sm"
                        style={{ left: `${thresh * 100}%` }}
                        title={`Tuned Decision Threshold: ${(thresh * 100).toFixed(0)}%`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Compare All Models View (if enabled) */}
          {compareMode && compareData && (
            <div className="glass-panel p-6 rounded-2xl shadow-xl overflow-x-auto">
              <h3 className="font-heading font-bold text-md text-white mb-4 flex items-center gap-2">
                <Layers className="w-4 h-4 text-purple-400" />
                Cross-Algorithm Predictions Comparison
              </h3>
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2 pr-4 font-semibold">Model</th>
                    <th className="py-2 px-2 font-semibold text-center text-amber-400">Fork</th>
                    <th className="py-2 px-2 font-semibold text-center text-emerald-400">Pin</th>
                    <th className="py-2 px-2 font-semibold text-center text-cyan-400">Skewer</th>
                    <th className="py-2 px-2 font-semibold text-center text-purple-400">Disc. Att.</th>
                    <th className="py-2 pl-2 font-semibold text-right">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {Object.entries(compareData.predictions).map(([mKey, mPred]) => (
                    <tr key={mKey} className="hover:bg-slate-800/30">
                      <td className="py-2.5 pr-4 font-sans font-medium text-slate-200">
                        {mPred.model_name}
                      </td>
                      {['fork', 'pin', 'skewer', 'discoveredAttack'].map(motif => {
                        const prob = mPred.probabilities[motif as keyof typeof mPred.probabilities] ?? 0;
                        const thresh = mPred.thresholds[motif as keyof typeof mPred.thresholds] ?? 0.5;
                        const detected = prob >= thresh;
                        return (
                          <td key={motif} className="py-2.5 px-2 text-center">
                            <span
                              className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                                detected
                                  ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                                  : 'text-slate-400'
                              }`}
                            >
                              {(prob * 100).toFixed(0)}%
                            </span>
                          </td>
                        );
                      })}
                      <td className="py-2.5 pl-2 text-right text-slate-400">
                        {mPred.latency_ms.toFixed(1)} ms
                      </td>
                    </tr>
                  ))}
                  {/* Rule Baseline Row */}
                  <tr className="bg-slate-900/60 font-medium">
                    <td className="py-2.5 pr-4 text-slate-300 font-sans">Rule-Based Detector (Baseline)</td>
                    {['fork', 'pin', 'skewer', 'discoveredAttack'].map(motif => {
                      const detected = compareData.rule_baseline.flags[motif] === 1;
                      return (
                        <td key={motif} className="py-2.5 px-2 text-center">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                              detected
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                : 'text-slate-500'
                            }`}
                          >
                            {detected ? 'YES' : 'NO'}
                          </span>
                        </td>
                      );
                    })}
                    <td className="py-2.5 pl-2 text-right text-slate-400">0.5 ms</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {/* Feature Attribution & Explanation Panel */}
          {explanation && explanation.top_features.length > 0 && (
            <div className="glass-panel p-6 rounded-2xl shadow-xl">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <h3 className="font-heading font-bold text-md text-white flex items-center gap-2">
                  <HelpCircle className="w-4 h-4 text-indigo-400" />
                  Key Features Driving This Prediction
                </h3>
                <span className="text-xs text-slate-400">SHAP / Tree Attribution</span>
              </div>
              <div className="mt-4 space-y-2.5">
                {explanation.top_features.map((feat, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/80 text-xs"
                  >
                    <div className="flex items-center space-x-2">
                      <span className="w-5 h-5 rounded-full bg-slate-800 flex items-center justify-center font-mono text-[10px] text-slate-400">
                        {idx + 1}
                      </span>
                      <span className="font-mono text-slate-300">{feat.feature}</span>
                    </div>
                    <div className="flex items-center space-x-3 font-mono">
                      <span className="text-slate-400">val: {feat.value}</span>
                      <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 font-bold border border-blue-500/20">
                        +{feat.importance}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
