import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Trophy, Clock, Target, Layers } from 'lucide-react';
import { apiClient } from '../api/client';

export const Comparison: React.FC = () => {
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [selectedModelCM, setSelectedModelCM] = useState<string>('random_forest');

  useEffect(() => {
    apiClient
      .getMetrics()
      .then(data => {
        setMetrics(data);
        const keys = Object.keys(data).filter(k => !k.startsWith('baseline'));
        if (keys.length > 0) setSelectedModelCM(keys[0]);
      })
      .catch(err => console.error(err));
  }, []);

  // Format data for comparison table & charts
  const tableData = Object.entries(metrics).map(([key, m]) => ({
    key,
    name: m.name || key.replace('_', ' ').toUpperCase(),
    macro_f1: m.macro_f1,
    micro_f1: m.micro_f1,
    subset_accuracy: m.subset_accuracy,
    hamming_loss: m.hamming_loss,
    jaccard_score: m.jaccard_score,
    train_time: m.training_time_sec ?? 0,
    inf_time: m.inference_time_ms ?? 0,
    fork_f1: m.per_label?.fork?.f1 ?? 0,
    pin_f1: m.per_label?.pin?.f1 ?? 0,
    skewer_f1: m.per_label?.skewer?.f1 ?? 0,
    disc_f1: m.per_label?.discoveredAttack?.f1 ?? 0,
  }));

  // Best model calculations
  const bestF1Model = tableData.reduce((prev, curr) => (curr.macro_f1 > (prev?.macro_f1 ?? 0) ? curr : prev), tableData[0]);
  const bestSubsetModel = tableData.reduce((prev, curr) => (curr.subset_accuracy > (prev?.subset_accuracy ?? 0) ? curr : prev), tableData[0]);
  const fastestModel = tableData.filter(m => m.inf_time > 0).reduce((prev, curr) => (curr.inf_time < (prev?.inf_time ?? 999) ? curr : prev), tableData[0]);

  // Chart data: per-label F1 scores across models
  const chartData = tableData.filter(d => !d.key.startsWith('baseline_dummy')).map(d => ({
    name: d.name.length > 12 ? d.name.substring(0, 11) + '..' : d.name,
    Fork: parseFloat((d.fork_f1 * 100).toFixed(1)),
    Pin: parseFloat((d.pin_f1 * 100).toFixed(1)),
    Skewer: parseFloat((d.skewer_f1 * 100).toFixed(1)),
    'Disc. Att.': parseFloat((d.disc_f1 * 100).toFixed(1)),
  }));

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold font-heading text-white tracking-tight flex items-center gap-2">
          <Trophy className="w-8 h-8 text-amber-400" />
          Model Evaluation & Performance Benchmarks
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Rigorous held-out test set metrics across classical, neural, and ensemble architectures.
        </p>
      </div>

      {/* Metric Callout Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-blue-500 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
            <span>Top Macro F1</span>
            <Trophy className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-heading text-white">
            {(bestF1Model?.macro_f1 * 100 || 0).toFixed(1)}%
          </div>
          <div className="text-xs text-blue-400 font-medium mt-1 truncate">
            {bestF1Model?.name || 'N/A'}
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-emerald-500 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
            <span>Subset Accuracy</span>
            <Target className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-heading text-white">
            {(bestSubsetModel?.subset_accuracy * 100 || 0).toFixed(1)}%
          </div>
          <div className="text-xs text-emerald-400 font-medium mt-1 truncate">
            Exact match: {bestSubsetModel?.name || 'N/A'}
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-purple-500 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
            <span>Lowest Latency</span>
            <Clock className="w-4 h-4 text-purple-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-heading text-white">
            {(fastestModel?.inf_time || 0).toFixed(2)} ms
          </div>
          <div className="text-xs text-purple-400 font-medium mt-1 truncate">
            Fastest: {fastestModel?.name || 'N/A'}
          </div>
        </div>

        <div className="glass-panel p-5 rounded-2xl border-l-4 border-l-cyan-500 shadow-lg">
          <div className="flex items-center justify-between text-slate-400 text-xs font-semibold uppercase tracking-wider">
            <span>Models Benchmarked</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="mt-2 text-2xl font-bold font-heading text-white">
            {tableData.length}
          </div>
          <div className="text-xs text-slate-400 font-medium mt-1">
            Classical, Deep Learning & Ensemble
          </div>
        </div>
      </div>

      {/* Main Leaderboard Table */}
      <div className="glass-panel rounded-2xl shadow-xl overflow-hidden border border-slate-800">
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <h2 className="font-heading font-bold text-lg text-white">
            Algorithm Comparison Leaderboard
          </h2>
          <span className="text-xs text-slate-400 font-mono">Held-out 15% Stratified Test Split</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800 font-semibold">
              <tr>
                <th className="py-3 px-4">Model Algorithm</th>
                <th className="py-3 px-3 text-center">Macro F1</th>
                <th className="py-3 px-3 text-center">Micro F1</th>
                <th className="py-3 px-3 text-center">Subset Acc.</th>
                <th className="py-3 px-3 text-center">Hamming Loss</th>
                <th className="py-3 px-3 text-center">Jaccard</th>
                <th className="py-3 px-3 text-right">Train Time</th>
                <th className="py-3 px-4 text-right">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {tableData.map(row => {
                const isTop = row.key === bestF1Model?.key;
                const isBaseline = row.key.startsWith('baseline');

                return (
                  <tr
                    key={row.key}
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isTop ? 'bg-blue-600/10 font-bold' : isBaseline ? 'bg-slate-950/40 text-slate-400' : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-slate-200 flex items-center gap-2">
                      {row.name}
                      {isTop && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40">
                          TOP
                        </span>
                      )}
                      {isBaseline && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-slate-400">
                          BASELINE
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-center font-bold text-blue-400">
                      {(row.macro_f1 * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-center text-slate-300">
                      {(row.micro_f1 * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-center text-emerald-400 font-semibold">
                      {(row.subset_accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-center text-slate-300">
                      {row.hamming_loss.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-center text-slate-300">
                      {row.jaccard_score.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-400">
                      {row.train_time.toFixed(1)}s
                    </td>
                    <td className="py-3 px-4 text-right text-slate-300">
                      {row.inf_time.toFixed(2)} ms
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Per-Motif F1 Bar Chart */}
      <div className="glass-panel p-6 rounded-2xl shadow-xl space-y-4">
        <div>
          <h3 className="font-heading font-bold text-lg text-white">
            Per-Motif F1 Score Comparison (%)
          </h3>
          <p className="text-xs text-slate-400">
            Compare how accurately each model recognizes Forks, Pins, Skewers, and Discovered Attacks.
          </p>
        </div>

        <div className="h-[380px] w-full pt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#94a3b8" angle={-20} textAnchor="end" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" domain={[0, 100]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '12px',
                  fontSize: '12px',
                }}
              />
              <Legend verticalAlign="top" height={36} />
              <Bar dataKey="Fork" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Pin" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Skewer" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Disc. Att." fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrices Inspector */}
      {metrics[selectedModelCM] && (
        <div className="glass-panel p-6 rounded-2xl shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
            <div>
              <h3 className="font-heading font-bold text-lg text-white">
                Confusion Matrices Inspector
              </h3>
              <p className="text-xs text-slate-400">
                True vs Predicted binary counts for each tactical motif on the held-out test set.
              </p>
            </div>
            <select
              value={selectedModelCM}
              onChange={e => setSelectedModelCM(e.target.value)}
              className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-sm font-medium text-white focus:outline-none"
            >
              {Object.keys(metrics).map(mKey => (
                <option key={mKey} value={mKey}>
                  {metrics[mKey].name || mKey}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
            {['fork', 'pin', 'skewer', 'discoveredAttack'].map(motifKey => {
              const labelData = metrics[selectedModelCM]?.per_label?.[motifKey];
              if (!labelData) return null;
              const cm = labelData.confusion_matrix;

              return (
                <div key={motifKey} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between text-xs font-semibold text-white uppercase tracking-wider">
                    <span>{motifKey}</span>
                    <span className="font-mono text-blue-400">F1: {(labelData.f1 * 100).toFixed(1)}%</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono pt-2">
                    <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
                      <div className="text-[10px] text-slate-400 uppercase">True Pos (TP)</div>
                      <div className="text-base font-bold mt-0.5">{cm.tp}</div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-300">
                      <div className="text-[10px] text-slate-400 uppercase">False Pos (FP)</div>
                      <div className="text-base font-bold mt-0.5">{cm.fp}</div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300">
                      <div className="text-[10px] text-slate-400 uppercase">False Neg (FN)</div>
                      <div className="text-base font-bold mt-0.5">{cm.fn}</div>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
                      <div className="text-[10px] text-slate-400 uppercase">True Neg (TN)</div>
                      <div className="text-base font-bold mt-0.5">{cm.tn}</div>
                    </div>
                  </div>

                  <div className="text-[11px] text-slate-400 flex justify-between pt-1">
                    <span>ROC-AUC: {(labelData.roc_auc * 100).toFixed(1)}%</span>
                    <span>PR-AUC: {(labelData.pr_auc * 100).toFixed(1)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
