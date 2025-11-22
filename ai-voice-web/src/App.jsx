import React, { useState } from "react";
import "./index.css";

const API_URL = "https://vox-dz3z.onrender.com/analyze";

const CHALLENGE_PHRASES = [
  "Hold position at checkpoint",
  "Authentication required for override",
  "Alpha team proceed to sector three",
  "Secure channel only, confirm identity",
  "Abort mission on my signal",
  "Confirm identity and continue",
  "Standby for encrypted transmission",
  "Command authority verification required",
  "Do not execute until code is verified",
  "Report status from sector seven immediately",
];

// Example files under /public/examples/...
const EXAMPLE_MAP = {
  none: null,
  "real-enrolled": {
    label: "Real: enrolled-like",
    path: "/examples/real_enrolled.wav",
  },
  "fake-tts": {
    label: "Fake: TTS-like",
    path: "/examples/fake_tts.wav",
  },
};

function App() {
  // --- state ---
  const [siteTab, setSiteTab] = useState("demo"); // demo | how | about

  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const [juryMode, setJuryMode] = useState(true);
  const [similarityThr, setSimilarityThr] = useState(0.8);
  const [anomalyThr, setAnomalyThr] = useState(0.0);

  const [challengeEnabled, setChallengeEnabled] = useState(true);
  const [challengePhrase, setChallengePhrase] = useState("");

  const [exampleKey, setExampleKey] = useState("none");

  const [tab, setTab] = useState("quick"); // quick | expert
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  // zoom for spectrogram
  const [zoom, setZoom] = useState(1.0); // 1.0 = 100%

  // --- file handling ---
  const onFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    if (f) {
      const url = URL.createObjectURL(f);
      setAudioUrl(url);
    } else {
      setAudioUrl(null);
    }
  };

  // load example from /public/examples/...
  const handleLoadExample = async () => {
    const ex = EXAMPLE_MAP[exampleKey];
    if (!ex) {
      alert("Choose an example type first.");
      return;
    }
    try {
      const res = await fetch(ex.path);
      if (!res.ok) throw new Error("Example not found");
      const blob = await res.blob();
      const exampleFile = new File([blob], ex.path.split("/").pop(), {
        type: blob.type,
      });
      setFile(exampleFile);
      setAudioUrl(URL.createObjectURL(blob));
    } catch (err) {
      console.error(err);
      alert(
        "Could not load example audio. Make sure files exist in public/examples/real_enrolled.wav and fake_tts.wav"
      );
    }
  };

  const handleGeneratePhrase = () => {
    const idx = Math.floor(Math.random() * CHALLENGE_PHRASES.length);
    setChallengePhrase(CHALLENGE_PHRASES[idx]);
  };

  // --- call backend ---
  const handleAnalyze = async () => {
    if (!file) {
      alert("Select or load an audio file first.");
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(API_URL, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const probReal = Number(data.prob_real ?? 0);
      const probFake = Number(data.prob_fake ?? 0);
      const backendVerdict = (data.verdict || "").toLowerCase();
      const isoScore = data.iso_score;
      const similarity = data.similarity;

      // decision stack
      let reasons = [];
      if (probReal < 0.6)
        reasons.push("Spoof model: low real probability");
      if (isoScore !== null && isoScore < anomalyThr)
        reasons.push("Anomaly detector: unusual embedding");
      if (
        data.has_enrollment &&
        similarity !== null &&
        similarity < similarityThr
      )
        reasons.push("Speaker mismatch: low similarity");

      let baseVerdict = "ACCEPT";
      if (backendVerdict === "fake" && probFake > 0.6) baseVerdict = "REJECT";

      let finalVerdict;
      if (baseVerdict === "ACCEPT" && reasons.length === 0)
        finalVerdict = "ACCEPT";
      else if (baseVerdict === "REJECT") finalVerdict = "REJECT";
      else finalVerdict = "SUSPICIOUS";

      // --- risk scoring ---
      // --- risk scoring ---

      // Base risk from prob_real
      let riskScore = 0;
      if (probReal >= 0.75) {
        riskScore += 0; // clearly human-ish
      } else if (probReal >= 0.5) {
        riskScore += 1; // borderline zone
      } else {
        riskScore += 2; // likely fake / untrustworthy
      }

      // Anomaly penalty
      if (isoScore !== null && isoScore < anomalyThr) {
        riskScore += 1;
      }

      // Speaker mismatch penalty
      if (data.has_enrollment && similarity !== null && similarity < similarityThr) {
        riskScore += 1;
      }

      let riskString;

      // HARD RULE: if prob_real < 0.5 → always HIGH
      if (probReal < 0.5) {
        riskString = "High";
      } else {
        // normal mapping for human-ish / borderline
        if (riskScore <= 1) {
          riskString = "Low";
        } else if (riskScore === 2) {
          riskString = "Medium";
        } else {
          riskString = "High";
        }
      }



      const merged = {
        ...data,
        prob_real: probReal,
        prob_fake: probFake,
        verdict_frontend: finalVerdict,
        risk_level_frontend: riskString,
        reasons,
        file_name: file.name,
        similarity_threshold_used: similarityThr,
        anomaly_threshold_used: anomalyThr,
      };

      setResult(merged);
      setHistory((prev) => [
        {
          file: file.name,
          verdict: finalVerdict,
          probReal,
          probFake,
          risk: riskString,
        },
        ...prev,
      ]);
    } catch (err) {
      console.error(err);
      alert("Error calling backend: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // download history as CSV
  const handleDownloadHistory = () => {
    if (!history.length) {
      alert("No history to download.");
      return;
    }
    const header = ["index", "file", "verdict", "prob_real", "prob_fake", "risk"];
    const rows = history.map((h, idx) => [
      history.length - idx,
      h.file,
      h.verdict,
      h.probReal.toFixed(3),
      h.probFake.toFixed(3),
      h.risk,
    ]);
    const csv = [header.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "voice_auth_session_history.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const verdictStyle = (() => {
    if (!result) return { cls: "", label: "" };
    const v = result.verdict_frontend;
    if (v === "ACCEPT") return { cls: "pill-real", label: "AUTHENTIC" };
    if (v === "REJECT") return { cls: "pill-fake", label: "SUSPECTED SPOOF" };
    return { cls: "pill-susp", label: "SUSPICIOUS" };
  })();

  // ---------------- RENDER ----------------
  return (
    <div className="site-root">
      {/* Global header */}
      <header className="topbar">
      <div className="brand">
        <div className="brand-icon brand-icon-shield">
          <svg
            viewBox="0 0 64 64"
            className="brand-shield-svg"
            aria-hidden="true"
          >
            {/* Outer shield */}
            <path
              d="M32 4 L50 10 L50 28 C50 40 42 50 32 56 C22 50 14 40 14 28 L14 10 Z"
              className="shield-outline"
            />
            {/* Inner glow */}
            <path
              d="M32 8 L46 13 L46 27 C46 37 40 45 32 50 C24 45 18 37 18 27 L18 13 Z"
              className="shield-inner"
            />
            {/* Center vertical spine */}
            <path
              d="M32 12 L32 44"
              className="shield-spine"
            />
            {/* Three 'heads' of Cerberus */}
            <circle cx="24" cy="24" r="3" className="cerb-head" />
            <circle cx="32" cy="22" r="3" className="cerb-head" />
            <circle cx="40" cy="24" r="3" className="cerb-head" />
          </svg>
        </div>
        <div>
          <div className="brand-title">Cerbervox</div>
          <div className="brand-sub">Only real voices get through</div>
        </div>
      </div>


        <nav className="topnav">
          <button
            className={`topnav-item ${siteTab === "demo" ? "active" : ""}`}
            onClick={() => setSiteTab("demo")}
          >
            Live demo
          </button>
          <button
            className={`topnav-item ${siteTab === "how" ? "active" : ""}`}
            onClick={() => setSiteTab("how")}
          >
            How it works
          </button>
          <button
            className={`topnav-item ${siteTab === "about" ? "active" : ""}`}
            onClick={() => setSiteTab("about")}
          >
            Use cases
          </button>
        </nav>
      </header>

      {siteTab === "demo" && (
        <div className="app-shell">
          {/* CONTROL RAIL */}
          <aside className="rail">
            <div className="rail-section">
              <div className="rail-title">Session settings</div>
              <label className="rail-toggle">
                <input
                  type="checkbox"
                  checked={juryMode}
                  onChange={(e) => setJuryMode(e.target.checked)}
                />
                <span>Demo mode (simplified)</span>
              </label>
            </div>

            <div className="rail-section">
              <div className="rail-title">Decision thresholds</div>

              <div className="rail-slider">
                <div className="rail-slider-label">
                  Speaker similarity (cosine)
                  <span>{similarityThr.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.6"
                  max="0.99"
                  step="0.01"
                  value={similarityThr}
                  onChange={(e) =>
                    setSimilarityThr(parseFloat(e.target.value))
                  }
                />
                <p className="rail-hint">
                  Higher → stricter identity match (fewer impostors, more false
                  rejections).
                </p>
              </div>

              <div className="rail-slider">
                <div className="rail-slider-label">
                  Anomaly score threshold
                  <span>{anomalyThr.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="-1.5"
                  max="1.5"
                  step="0.05"
                  value={anomalyThr}
                  onChange={(e) => setAnomalyThr(parseFloat(e.target.value))}
                />
                <p className="rail-hint">
                  Lower → detector flags more samples as anomalous / risky.
                </p>
              </div>
            </div>

            <div className="rail-section">
              <label className="rail-toggle">
                <input
                  type="checkbox"
                  checked={challengeEnabled}
                  onChange={(e) => setChallengeEnabled(e.target.checked)}
                />
                <span>Enable challenge-response concept</span>
              </label>
            </div>

            <div className="rail-section">
              <div className="rail-title">Demo examples</div>
              <select
                className="rail-select"
                value={exampleKey}
                onChange={(e) => setExampleKey(e.target.value)}
              >
                <option value="none">None selected</option>
                <option value="real-enrolled">Real: enrolled-like</option>
                <option value="fake-tts">Fake: TTS / cloned</option>
              </select>
              <button
                className="btn secondary full"
                onClick={handleLoadExample}
              >
                Load selected example
              </button>
            </div>

            <div className="rail-section">
              <div className="rail-title">Model status</div>
              <p className="rail-hint">
                Enrolled identities:{" "}
                {result?.has_enrollment ? "available" : "0 (demo)"}
              </p>
            </div>
          </aside>

          {/* MAIN CONSOLE */}
          <main className="console">
            <section className="hero">
              <div>
                <h1>AI-Based Voice Authentication & Anti-Deepfake Console</h1>
                <p>
                  This prototype simulates a secure voice gate in front of
                  critical communication channels. Audio is inspected for
                  spoofing, anomalies and identity mismatch before being allowed
                  through.
                </p>
                <ul>
                  <li>Spoof classifier → real vs synthetic / cloned voice</li>
                  <li>
                    Anomaly detector → unusual embeddings vs known real speech
                  </li>
                  <li>
                    Speaker matching (optional) → cosine similarity vs enrolled
                    voices
                  </li>
                  <li>
                    Policy engine → ACCEPT / REJECT / SUSPICIOUS + risk level
                  </li>
                </ul>
              </div>
            </section>

            <div className="tabs">
              <button
                className={`tab ${tab === "quick" ? "active" : ""}`}
                onClick={() => setTab("quick")}
              >
                ⚡ Quick verdict
              </button>
              <button
                className={`tab ${tab === "expert" ? "active" : ""}`}
                onClick={() => setTab("expert")}
              >
                🧠 Expert analysis
              </button>
            </div>

            {/* QUICK VERDICT VIEW */}
            {tab === "quick" && (
              <section className="grid-two">
                {/* input card */}
                <div className="card">
                  <div className="card-header">
                    <h2>1. Provide audio</h2>
                    <span className="badge">
                      WAV / MP3 / FLAC / OGG / M4A
                    </span>
                  </div>
                  <p className="card-sub">
                    Upload a field recording, a normal human clip, or a TTS /
                    cloned sample to see how the gate responds.
                  </p>

                  <div className="input-row">
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={onFileChange}
                      className="file-input"
                    />
                    <button
                      className="btn primary"
                      onClick={handleAnalyze}
                      disabled={loading}
                    >
                      {loading ? "Analyzing…" : "▶ Analyze sample"}
                    </button>
                  </div>

                  {audioUrl && (
                    <div className="player">
                      <audio controls src={audioUrl} />
                    </div>
                  )}

                  {challengeEnabled && (
                    <div className="challenge">
                      <div className="challenge-top">
                        <div>
                          <div className="challenge-label">
                            Challenge–response phrase (concept)
                          </div>
                          <p>
                            In a real system, the operator issues a one-time
                            phrase. Caller must repeat it live to bind the
                            command to a human.
                          </p>
                        </div>
                        <button
                          className="btn ghost"
                          onClick={handleGeneratePhrase}
                        >
                          Generate phrase
                        </button>
                      </div>
                      {challengePhrase && (
                        <div className="challenge-phrase">
                          {challengePhrase}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* verdict card */}
                <div className="card">
                  <div className="card-header">
                    <h2>2. Verdict & risk</h2>
                    <span className="badge accent">
                      {result ? "Sample analyzed" : "Awaiting input"}
                    </span>
                  </div>

                  {!result && (
                    <div className="placeholder">
                      Upload or load an example, then hit{" "}
                      <span className="mono">Analyze sample</span>.
                    </div>
                  )}

                  {result && (
                    <>
                      <div
                        className={`verdict ${
                          result.verdict_frontend === "ACCEPT"
                            ? "verdict-accept"
                            : result.verdict_frontend === "REJECT"
                            ? "verdict-reject"
                            : "verdict-suspicious"
                        }`}
                      >
                        <div className="verdict-main">
                          <div className={`pill ${verdictStyle.cls}`}>
                            {verdictStyle.label}
                          </div>
                          <div className="verdict-title">
                            {result.verdict_frontend === "ACCEPT"
                              ? "Access granted – voice accepted"
                              : result.verdict_frontend === "REJECT"
                              ? "Access denied – spoof suspected"
                              : "Proceed with caution – suspicious sample"}
                          </div>
                          <div className="verdict-file">
                            File: <code>{result.file_name}</code>
                          </div>
                        </div>

                        <div className="prob-panel">
                          <div className="prob-gauge real">
                            <div className="prob-value">
                              {(result.prob_real * 100).toFixed(0)}%
                            </div>
                            <div className="prob-label">REAL</div>
                          </div>
                          <div className="prob-gauge fake">
                            <div className="prob-value">
                              {(result.prob_fake * 100).toFixed(0)}%
                            </div>
                            <div className="prob-label">FAKE</div>
                          </div>
                                                    <div
                            className={`risk-chip risk-${result.risk_level_frontend
                              .toLowerCase()
                              .replace(/\s+/g, "-")}`}
                          >
                            <div className="risk-label">Risk</div>
                            <div className="risk-value">{result.risk_level_frontend}</div>
                          </div>

                        </div>
                      </div>

                      {!juryMode && result.reasons?.length > 0 && (
                        <ul className="reason-list">
                          {result.reasons.map((r, i) => (
                            <li key={i}>{r}</li>
                          ))}
                        </ul>
                      )}

                      <div className="history-block">
                        <div className="history-header-row">
                          <h3>Session history</h3>
                          <button
                            className="btn ghost small"
                            onClick={handleDownloadHistory}
                          >
                            ⬇ Download CSV
                          </button>
                        </div>
                        {history.length === 0 ? (
                          <div className="placeholder small">
                            No samples analyzed yet in this browser session.
                          </div>
                        ) : (
                          <div className="history-table-wrap">
                            <table className="history-table">
                              <thead>
                                <tr>
                                  <th>#</th>
                                  <th>File</th>
                                  <th>Verdict</th>
                                  <th>Prob R</th>
                                  <th>Prob F</th>
                                  <th>Risk</th>
                                </tr>
                              </thead>
                              <tbody>
                                {history.map((h, i) => (
                                  <tr key={i}>
                                    <td>{history.length - i}</td>
                                    <td>{h.file}</td>
                                    <td>
                                      <span
                                        className={`tag ${
                                          h.verdict === "ACCEPT"
                                            ? "tag-real"
                                            : h.verdict === "REJECT"
                                            ? "tag-fake"
                                            : "tag-susp"
                                        }`}
                                      >
                                        {h.verdict}
                                      </span>
                                    </td>
                                    <td>{h.probReal.toFixed(2)}</td>
                                    <td>{h.probFake.toFixed(2)}</td>
                                    <td>{h.risk}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </section>
            )}

            {/* EXPERT VIEW */}
            {tab === "expert" && (
              <section className="grid-two">
                {!result ? (
                  <div className="card">
                    <h2>Expert analysis</h2>
                    <div className="placeholder">
                      Run a sample in <span className="mono">Quick verdict</span>{" "}
                      first.
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="card">
                      <div className="card-header">
                        <h2>Waveform & spectrogram</h2>
                        <div className="zoom-toolbar">
                          <span>Zoom</span>
                          <button
                            className="zoom-btn"
                            onClick={() =>
                              setZoom((z) => Math.max(0.5, z - 0.25))
                            }
                          >
                            −
                          </button>
                          <span className="zoom-level">
                            {(zoom * 100).toFixed(0)}%
                          </span>
                          <button
                            className="zoom-btn"
                            onClick={() =>
                              setZoom((z) => Math.min(3, z + 0.25))
                            }
                          >
                            +
                          </button>
                          <button
                            className="zoom-btn reset"
                            onClick={() => setZoom(1.0)}
                          >
                            Reset
                          </button>
                        </div>
                      </div>
                      {result.plot_image ? (
                        <div className="plot-zoom-container">
                          <div
                            className="plot-zoom-inner"
                            style={{
                              transform: `scale(${zoom})`,
                              transformOrigin: "top left",
                            }}
                          >
                            <img
                              src={result.plot_image}
                              alt="Waveform and spectrogram"
                              className="plot-image"
                            />
                          </div>
                        </div>
                      ) : (
                        <div className="placeholder">
                          No plot image returned from backend.
                        </div>
                      )}
                    </div>
                    <div className="card">
                      <h2>Internal metrics</h2>
                      <ul className="metrics-list">
                        <li>Prob. real: {result.prob_real.toFixed(3)}</li>
                        <li>Prob. fake: {result.prob_fake.toFixed(3)}</li>
                        <li>
                          Anomaly score:{" "}
                          {result.iso_score !== null
                            ? result.iso_score.toFixed(3)
                            : "None"}
                        </li>
                        <li>
                          Risk (frontend): {result.risk_level_frontend}
                        </li>
                        <li>
                          Best match:{" "}
                          {result.best_match !== null
                            ? result.best_match
                            : "None"}
                        </li>
                        <li>
                          Similarity:{" "}
                          {result.similarity !== null
                            ? result.similarity.toFixed(3)
                            : "None"}
                        </li>
                      </ul>

                      {!juryMode && (
                        <div className="raw-json">
                          <div className="raw-json-title">
                            Raw JSON (debug)
                          </div>
                          <pre>{JSON.stringify(result, null, 2)}</pre>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </section>
            )}
          </main>
        </div>
      )}

      {siteTab === "how" && (
        <main className="static-main">
          <section className="static-card">
            <h2>How Cerbervox evaluates audio</h2>
            <ol>
              <li>
                <b>Ingest & normalize.</b> The uploaded signal is resampled,
                normalized, and trimmed to a usable segment.
              </li>
              <li>
                <b>Feature extraction.</b> MFCCs and spectral statistics are
                computed to represent the voice in a compact numerical form.
              </li>
              <li>
                <b>Spoof classifier.</b> A RandomForest model predicts whether
                the clip is more likely genuine human speech or spoofed /
                synthetic.
              </li>
              <li>
                <b>Anomaly detector.</b> An IsolationForest checks if the
                embedding lies inside the manifold of known real voices.
              </li>
              <li>
                <b>Identity check (optional).</b> Cosine similarity compares the
                sample to enrolled voiceprints.
              </li>
              <li>
                <b>Policy engine.</b> Scores are fused into the final verdict
                and risk level you see in the console.
              </li>
            </ol>
          </section>
        </main>
      )}

      {siteTab === "about" && (
        <main className="static-main">
          <section className="static-card">
            <h2>Use cases & positioning</h2>
            <p>
              Cerbervox is a prototype exploring how AI can protect voice
              channels from deepfake and spoofed audio. It is designed as a{" "}
              <b>front-door filter</b> before high-impact actions.
            </p>
            <ul>
              <li>Pre-screening suspicious calls in security / ops centers</li>
              <li>
                Guarding remote command channels and “voice-only” escalation
                paths
              </li>
              <li>Training / demo platform for cyber & forensics teams</li>
              <li>Educational sandbox for anti-deepfake research</li>
            </ul>
            <p>
              This implementation is not field-grade yet, but the architecture
              mirrors what a hardened system would use: multi-signal scoring,
              challenge-response concepts, and explicit risk surfacing for human
              operators.
            </p>
          </section>
        </main>
        
      )}
      <Footer setSiteTab={setSiteTab} />

    </div>
  );
}

function Footer({ setSiteTab }) {
  return (
    <footer className="footer">
      <div className="footer-main">
        {/* Brand + description */}
        <div className="footer-column">
          <div className="footer-brand">
            <div className="brand-icon brand-icon-shield footer-icon">
              {/* reuse existing shield SVG */}
              <svg
                viewBox="0 0 64 64"
                className="brand-shield-svg"
                aria-hidden="true"
              >
                <path
                  d="M32 4 L50 10 L50 28 C50 40 42 50 32 56 C22 50 14 40 14 28 L14 10 Z"
                  className="shield-outline"
                />
                <path
                  d="M32 8 L46 13 L46 27 C46 37 40 45 32 50 C24 45 18 37 18 27 L18 13 Z"
                  className="shield-inner"
                />
                <path d="M32 12 L32 44" className="shield-spine" />
                <circle cx="24" cy="24" r="3" className="cerb-head" />
                <circle cx="32" cy="22" r="3" className="cerb-head" />
                <circle cx="40" cy="24" r="3" className="cerb-head" />
              </svg>
            </div>
            <div>
              <div className="footer-title">Cerbervox</div>
              <div className="footer-sub">
                AI voice gate that blocks deepfake and spoofed audio before it
                reaches critical channels.
              </div>
            </div>
          </div>
        </div>

        {/* Nav links */}
        <div className="footer-column">
          <div className="footer-heading">Navigate</div>
          <button
            className="footer-link"
            onClick={() => setSiteTab("demo")}
          >
            Live demo
          </button>
          <button
            className="footer-link"
            onClick={() => setSiteTab("how")}
          >
            How it works
          </button>
          <button
            className="footer-link"
            onClick={() => setSiteTab("about")}
          >
            Use cases
          </button>
          <a
            className="footer-link"
            href="https://github.com/abhigyanshrivastav/vox"
            target="_blank"
            rel="noreferrer"
          >
            GitHub repo
          </a>
        </div>

        {/* Status + disclaimer */}
        <div className="footer-column">
          <div className="footer-heading">Prototype status</div>
          <p className="footer-text">
            This is an academic prototype for demonstrating anti-deepfake voice
            authentication. It is <b>not</b> certified or hardened for real
            military or commercial deployment.
          </p>
          <div className="footer-tags">
            <span className="footer-tag">Python</span>
            <span className="footer-tag">FastAPI</span>
            <span className="footer-tag">scikit-learn</span>
            <span className="footer-tag">React</span>
          </div>
        </div>
      </div>

      <div className="footer-bottom">
        <span>© 2025 Cerbervox · Academic demo</span>
        
      </div>
    </footer>
  );
}


export default App;
