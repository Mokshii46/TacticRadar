import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Analyzer } from './pages/Analyzer';
import { Comparison } from './pages/Comparison';
import { Insights } from './pages/Insights';
import { About } from './pages/About';

export function App() {
  const [activeTab, setActiveTab] = useState<'analyzer' | 'comparison' | 'insights' | 'about'>('analyzer');
  const [apiHealthy, setApiHealthy] = useState<boolean>(true);

  // Poll backend health status
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/api/health');
        setApiHealthy(res.ok);
      } catch {
        setApiHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col selection:bg-blue-600 selection:text-white">
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} apiHealthy={apiHealthy} />

      {/* Main View Area */}
      <main className="flex-1">
        {activeTab === 'analyzer' && <Analyzer />}
        {activeTab === 'comparison' && <Comparison />}
        {activeTab === 'insights' && <Insights />}
        {activeTab === 'about' && <About />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/60 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© 2026 TacticRadar — Multi-Label Chess Tactical Opportunity Detection.</p>
          <p className="flex items-center gap-2">
            <span>Data: Lichess Puzzle Database (CC0)</span>
            <span>•</span>
            <a
              href="https://database.lichess.org/#puzzles"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:underline"
            >
              Lichess.org
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
