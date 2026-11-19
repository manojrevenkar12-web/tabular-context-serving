/** Motion and layout constants that code (not CSS) needs. Colours stay in tokens.css. */
export const motion = {
  revealStaggerS: 0.045,
  revealS: 0.42,
  playStepMs: 450,
  ease: [0.2, 0.7, 0.2, 1] as const,
};

export const units = {
  bytesPerCell: 1e9,
  weightsBytes: 2.2e9,
};
