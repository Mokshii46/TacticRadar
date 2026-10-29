import React from 'react';
import { Database, Filter, GitBranch, Layers, ShieldAlert, BarChart2 } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export const Insights: React.FC = () => {
  // Label distribution data
  const labelDistData = [
    { name: 'Fork', count: 574, pct: 23.0, fill: '#f59e0b' },
    { name: 'Pin', count: 553, pct: 22.1, fill: '#10b981' },
    { name: 'Skewer', count: 508, pct: 20.3, fill: '#06b6d4' },
    { name: 'Discovered Attack', count: 552, pct: 22.1, fill: '#8b5cf6' },
    { name: 'Quiet (Negative Class)', count: 500, pct: 20.0, fill: '#64748b' },
  ];

  // Co-occurrence counts
  const cooccurrenceData = [
    { pair: 'Fork + Pin', count: 142 },
    { pair: 'Fork + Discovered Attack', count: 98 },
    { pair: 'Pin + Skewer', count: 64 },
    { pair: 'Pin + Discovered Attack', count: 112 },
    { pair: 'Skewer + Discovered Attack', count: 38 },
    { pair: 'Fork + Skewer', count: 46 },
  ];

  // Rating bracket distribution
  const ratingData = [
    { bracket: '< 1200 (Beginner)', count: 120 },
    { bracket: '1200-1500 (Intermediate)', count: 580 },
    { bracket: '1500-1800 (Advanced)', count: 940 },
    { bracket: '1800-2100 (Master)', count: 610 },
    { bracket: '2100+ (Grandmaster)', count: 250 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold font-heading text-white tracking-tight flex items-center gap-2">
          <Database className="w-8 h-8 text-blue-400" />
          Lichess Puzzle Dataset Insights & Pipeline
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Detailed breakdown of multi-label tactical motif distributions, negative quiet sampling, and FEN offset corrections.
        </p>
      </div>

      {/* Pipeline Architecture Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-2">
          <div className="flex items-center space-x-2 text-blue-400 font-semibold text-sm">
            <Filter className="w-4 h-4" />
            <span>1. Quality Filtering</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Filtered raw Lichess database of 4M+ puzzles to retain high-confidence entries with <strong>Popularity ≥ 70</strong> and <strong>NbPlays ≥ 50</strong>, eliminating ambiguous or poorly vetted positions.
          </p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-2">
          <div className="flex items-center space-x-2 text-emerald-400 font-semibold text-sm">
            <GitBranch className="w-4 h-4" />
            <span>2. FEN Offset Correction</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Raw Lichess FENs correspond to the state <em>before</em> the opponent's initiating move. Our pipeline applies <code>Moves[0]</code> with <code>python-chess</code> to derive the true puzzle starting position.
          </p>
        </div>

        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-2">
          <div className="flex items-center space-x-2 text-purple-400 font-semibold text-sm">
            <ShieldAlert className="w-4 h-4" />
            <span>3. Negative Quiet Sampling</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Every Lichess puzzle is inherently tactical. To prevent false positives in tranquil games, we sample <strong>20% quiet positions</strong> (tagged <code>quietMove</code>, <code>endgame</code>) with all-zero labels.
          </p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Label Distribution Chart */}
        <div className="glass-panel p-6 rounded-2xl shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-heading font-bold text-lg text-white flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-amber-400" />
              Motif Prevalence & Class Balance
            </h3>
            <span className="text-xs font-mono text-slate-400">Multi-label Multi-hot</span>
          </div>
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={labelDistData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#94a3b8" angle={-15} textAnchor="end" tick={{ fontSize: 11 }} />
                <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {labelDistData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Motif Co-occurrence Chart */}
        <div className="glass-panel p-6 rounded-2xl shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-heading font-bold text-lg text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              Tactical Motif Co-occurrence
            </h3>
            <span className="text-xs font-mono text-slate-400">Multi-label Overlaps</span>
          </div>
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cooccurrenceData} layout="vertical" margin={{ top: 10, right: 20, left: 60, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <YAxis dataKey="pair" type="category" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="count" fill="#818cf8" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Rating Bracket Breakdown */}
      <div className="glass-panel p-6 rounded-2xl shadow-xl space-y-4">
        <h3 className="font-heading font-bold text-lg text-white">
          Puzzle Difficulty Distribution (Glicko-2 Rating)
        </h3>
        <p className="text-xs text-slate-400">
          Distribution across tactical skill tiers, from beginner puzzles to master-level candidate moves.
        </p>
        <div className="h-[260px] w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ratingData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="bracket" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px', fontSize: '12px' }}
              />
              <Bar dataKey="count" fill="#3b82f6" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
