export interface SamplePuzzle {
  id: string;
  name: string;
  fen: string;
  turn: 'w' | 'b';
  expectedMotifs: string[];
  description: string;
}

export const SAMPLE_PUZZLES: SamplePuzzle[] = [
  {
    id: "fork_01",
    name: "Classic Royal Knight Fork",
    fen: "8/2q1k3/8/3N4/8/8/8/3K4 b - - 0 1",
    turn: "b",
    expectedMotifs: ["fork"],
    description: "White Knight on d5 simultaneously attacks the Black King on e7 and the Black Queen on c7."
  },
  {
    id: "pin_01",
    name: "Absolute Bishop Pin on the Queen",
    fen: "3k4/8/8/3q4/8/8/6B1/3K4 w - - 0 1",
    turn: "w",
    expectedMotifs: ["pin"],
    description: "White Bishop on g2 pins the Black Queen on d5 against the Black King on d8 along the long diagonal."
  },
  {
    id: "skewer_01",
    name: "Rook Skewer Through King to Queen",
    fen: "4q3/4k3/8/8/8/8/8/4R1K1 w - - 0 1",
    turn: "w",
    expectedMotifs: ["skewer"],
    description: "White Rook on e1 skewers the Black King on e7, forcing it to move and exposing the Queen on e8."
  },
  {
    id: "disc_01",
    name: "Discovered Attack Opening Queen Line",
    fen: "3q4/8/8/8/3N4/8/8/3R2K1 w - - 0 1",
    turn: "w",
    expectedMotifs: ["discoveredAttack"],
    description: "Moving the White Knight on d4 unmasks a direct vertical line of attack from the White Rook on d1 to the Black Queen on d8."
  },
  {
    id: "multi_01",
    name: "Double Tactic: Pin & Knight Fork",
    fen: "2r3k1/pp3ppp/8/3N4/8/8/PP3PPP/2R3K1 w - - 0 1",
    turn: "w",
    expectedMotifs: ["fork", "pin"],
    description: "Knight on d5 can deliver Ne7+ royal fork while Rook line creates pin tension."
  },
  {
    id: "quiet_01",
    name: "Quiet Balanced Position (Negative Sample)",
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    turn: "w",
    expectedMotifs: [],
    description: "Standard initial chess board. No forks, pins, skewers, or discovered attacks active."
  }
];
